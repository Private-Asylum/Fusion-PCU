//! A real 1,024-element loss and correctly normalized backward training graph.
//!
//! Exact dyadic witnesses isolate scheduling/storage and fault provenance from
//! vendor tolerance. This is correctness qualification, not a latency benchmark.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    dialect::tensor::{TensorError, TensorArithmeticStep},
};
use super::contract::POLICY_LOCK;

macro_rules! sources {
    ($scalar:ty, $train:ident, $loss:ident) => {
        #[pcu(flag(strict))]
        fn $train(
            input: &[[$scalar; 2]; 8],
            transpose: &[[$scalar; 8]; 2],
            weights: &[[$scalar; 128]; 2],
            target: &[[$scalar; 128]; 8],
            normalization: &[[$scalar; 128]; 8],
        ) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
            let preactivation = pcu::matmul(input, weights)?;
            let prediction = pcu::relu(&preactivation)?;
            let _loss = pcu::mean_squared_error(&prediction, target)?;
            let difference = pcu::sub(&prediction, target)?;
            // d(MSE)/d(prediction) = 2 * difference / 1,024. The factor
            // belongs before backward; folding it into SGD changes Strict order.
            let normalized = pcu::mul(&difference, normalization)?;
            let derivative = pcu::relu_backward(&preactivation, &normalized)?;
            let gradient = pcu::matmul(transpose, &derivative)?;
            pcu::sgd_update(weights, &gradient, 0.5)
        }

        #[pcu(flag(strict))]
        fn $loss(
            input: &[[$scalar; 2]; 8],
            weights: &[[$scalar; 128]; 2],
            target: &[[$scalar; 128]; 8],
        ) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
            let prediction = pcu::relu(pcu::matmul(input, weights)?)?;
            pcu::mean_squared_error(&prediction, target)
        }
    };
}
sources!(f32, train_f32, loss_f32);
sources!(f64, train_f64, loss_f64);

#[pcu]
fn retain<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

fn read<T: PcuScalar>(owner: &PcuTensor<T>, first: T, second: T, sentinel: T) {
    let mut stack = [sentinel; 260];
    owner.read_into(&mut stack).unwrap();
    for (actual, expected) in stack[..256]
        .iter()
        .zip([first; 128].iter().chain([second; 128].iter()))
    {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    for tail in &stack[256..] {
        assert_eq!(tail.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}

macro_rules! verify_format {
    ($scalar:ty, $train:ident, $loss:ident) => {{
        let input: [[$scalar; 2]; 8] = [[1.0, 0.0]; 8];
        let transpose: [[$scalar; 8]; 2] = [[1.0; 8], [0.0; 8]];
        let weights: [[$scalar; 128]; 2] = [[2.0; 128], [-1.0; 128]];
        let target: [[$scalar; 128]; 8] = [[1.0; 128]; 8];
        let normalization: [[$scalar; 128]; 8] = [[1.0 / 512.0; 128]; 8];
        let x = retain(&input).unwrap();
        let xt = retain(&transpose).unwrap();
        let w = retain(&weights).unwrap();
        let t = retain(&target).unwrap();
        let n = retain(&normalization).unwrap();
        let old = $train(&input, &transpose, &weights, &target, &normalization).unwrap();
        for role in 0..3 {
            let updated = match role {
                0 => $train(&input, &transpose, &weights, &target, &normalization),
                1 => $train(&x, &transpose, &w, &target, &n),
                _ => $train(&x, &xt, &w, &t, &n),
            }
            .unwrap();
            // Eight identical rows contribute 8/512 = 1/64 to each gradient.
            // Half-rate SGD changes each first-row weight to 2 - 1/128.
            read(&updated, 255.0 / 128.0, -1.0, 19.0);
            read(&old, 255.0 / 128.0, -1.0, 19.0);
        }
        let loss = $loss(&x, &w, &t).unwrap();
        assert!(loss.shape().is_empty());
        let mut scalar = [19.0; 3];
        loss.read_into(&mut scalar).unwrap();
        assert_eq!(
            scalar.map(<$scalar>::to_bits),
            [1.0, 19.0, 19.0].map(<$scalar>::to_bits)
        );

        let mut invalid = target;
        invalid[7][127] = <$scalar>::NAN;
        let error = $train(&x, &xt, &w, &invalid, &n).unwrap_err();
        assert!(
            matches!(
                error,
                PcuExecutionError::TensorBuild(TensorError::CompoundArithmeticFault {
                    element_index: 0,
                    reduction_index: 1023,
                    step: TensorArithmeticStep::Subtract,
                    kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                    ..
                })
            ),
            "lost discarded late loss provenance: {error:?}"
        );
        read(
            &$train(&x, &xt, &w, &t, &n).unwrap(),
            255.0 / 128.0,
            -1.0,
            19.0,
        );
        global::clear_thread_cache().unwrap();
        drop((x, xt, w, t, n));
        read(&old, 255.0 / 128.0, -1.0, 19.0);
        loss.read_into(&mut scalar).unwrap();
        assert_eq!(
            scalar.map(<$scalar>::to_bits),
            [1.0, 19.0, 19.0].map(<$scalar>::to_bits)
        );
    }};
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for float_underflow in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                global::configure(global::PcuExecutionPolicy {
                    backend,
                    float_underflow,
                    numerical_options: PcuNumericalOptions {
                        precision,
                        compound_arithmetic,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .unwrap();
                global::clear_thread_cache().unwrap();
                verify_format!(f32, train_f32, loss_f32);
                verify_format!(f64, train_f64, loss_f64);
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

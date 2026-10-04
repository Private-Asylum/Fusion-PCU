//! Ordinary annotated multi-stage graphs, with independent exact dyadic witnesses.
#[rustfmt::skip]
use fusion_pcu::{
    global, pcu, PcuScalar, PcuTensor, PcuExecutionError,
    PcuNumericalOptions, PcuPrecisionPolicy, PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy, PcuExecutionFaultKind,
    dialect::tensor::{TensorError, TensorArithmeticStep},
};
use super::{contract::POLICY_LOCK, source};

#[pcu(flag(strict))]
fn ordered<T: PcuScalar>(
    first: &[[T; 2]; 2],
    second: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _first = pcu::relu(first)?;
    let _second = pcu::relu(second)?;
    pcu::identity(first)
}
#[pcu]
fn retain<T: PcuScalar>(input: &[[T; 2]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu]
fn retain_column<T: PcuScalar>(input: &[[T; 1]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
fn read<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T], sentinel: T) {
    let mut stack = [sentinel; 6];
    owner.read_into(&mut stack).unwrap();
    for (a, b) in stack[..expected.len()].iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
    for tail in &stack[expected.len()..] {
        assert_eq!(tail.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}
macro_rules! format {
    ($scalar:ty, $train:ident, $loss:ident) => {{
        let input: [[$scalar; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
        let weights: [[$scalar; 1]; 2] = [[2.0], [-1.0]];
        let target: [[$scalar; 1]; 2] = [[1.0], [0.0]];
        let x = retain(&input).unwrap();
        let w = retain_column(&weights).unwrap();
        let t = retain_column(&target).unwrap();
        let old = source::$train(&input, &input, &weights, &target).unwrap();
        for role in 0..4 {
            let updated = match role {
                0 => source::$train(&input, &input, &weights, &target),
                1 => source::$train(&x, &input, &w, &target),
                2 => source::$train(&input, &x, &weights, &t),
                _ => source::$train(&x, &x, &w, &t),
            }
            .unwrap();
            read(&updated, &[1.5, -1.0], 19.0);
            read(&old, &[1.5, -1.0], 19.0);
        }
        let loss = source::$loss(&x, &w, &t).unwrap();
        assert!(loss.shape().is_empty());
        read(&loss, &[0.5], 19.0);
        // Loss is discarded in train, but its error is an observable effect.
        let invalid_target = [[1.0], [<$scalar>::NAN]];
        let error = source::$train(&x, &x, &w, &invalid_target).unwrap_err();
        let PcuExecutionError::TensorBuild(TensorError::CompoundArithmeticFault {
            element_index: 0,
            reduction_index: 1,
            step: TensorArithmeticStep::Subtract,
            kind: PcuExecutionFaultKind::InvalidFloatingOperand,
            ..
        }) = error
        else {
            panic!("lost discarded loss provenance: {error:?}")
        };
        let first = [[1.0, 2.0], [3.0, <$scalar>::NAN]];
        let second = [[<$scalar>::NAN, 2.0], [3.0, 4.0]];
        for (a, expected) in [(first, 3), (input, 0)] {
            let error = ordered(&a, &second).unwrap_err();
            let PcuExecutionError::TensorBuild(TensorError::ArithmeticFault {
                element_index,
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                ..
            }) = error
            else {
                panic!("lost ordered pointwise provenance: {error:?}")
            };
            assert_eq!(element_index, expected);
        }
        read(
            &ordered(&input, &input).unwrap(),
            &[1.0, 0.0, 0.0, 1.0],
            19.0,
        );
        // All failed calls are private: previously escaped payloads stay intact,
        // and clearing preparation cannot shorten their Rust-visible lifetime.
        global::clear_thread_cache().unwrap();
        drop(x);
        drop(w);
        drop(t);
        read(&old, &[1.5, -1.0], 19.0);
        read(&loss, &[0.5], 19.0);
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
                format!(f32, train_f32, loss_f32);
                format!(f64, train_f64, loss_f64);
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

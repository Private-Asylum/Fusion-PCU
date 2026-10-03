//! Requested AD paths, finite extremes, retained owners and original checked effects.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuScalar,
    PcuTensor,
};
use super::contract::POLICY_LOCK;

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(flag(strict))]
fn weight_gradient<T: PcuScalar>(
    input: &[T],
    weights: &[T],
    target: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let prediction = pcu::mul(input, weights);
    let loss = pcu::mean_squared_error(&prediction, target);
    pcu::gradient(&loss, weights)
}

#[pcu(flag(strict))]
fn input_gradient<T: PcuScalar>(
    input: &[T],
    weights: &[T],
    target: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let prediction = pcu::mul(input, weights);
    let loss = pcu::mean_squared_error(&prediction, target);
    pcu::gradient(&loss, input)
}

#[pcu(flag(strict))]
fn forward_unused_fault<T: PcuScalar>(
    input: &[T],
    weights: &[T],
    target: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _checked = pcu::div(weights, input);
    let prediction = pcu::mul(input, weights);
    let loss = pcu::mean_squared_error(&prediction, target);
    pcu::gradient(&loss, weights)
}

fn fault(result: Result<impl Sized, PcuExecutionError>, expected: PcuExecutionFaultKind) {
    match result {
        Err(PcuExecutionError::ArithmeticFault(fault)) => {
            assert_eq!(fault.kind, expected);
            assert_eq!(fault.invocation_id, 0);
            assert!(!fault.recovered);
        }
        Err(PcuExecutionError::TensorBuild(
            fusion_pcu::dialect::tensor::TensorError::ArithmeticFault {
                element_index,
                kind,
                ..
            },
        )) => {
            assert_eq!(kind, expected);
            assert_eq!(element_index, 0);
        }
        Err(error) => panic!("unexpected checked error: {error:?}"),
        Ok(_) => panic!("checked fault published a usable owner"),
    }
}

macro_rules! verify_format {
    ($scalar:ty) => {{
        let weights = retain(&[<$scalar>::MAX]).unwrap();
        let targets: [$scalar; 2] = [1.0, 0.5];
        let zero: $scalar = -0.0;
        let sentinel: $scalar = 99.0;
        for target in targets {
            let gradient = weight_gradient::<$scalar>(&[0.0], &weights, &[target]).unwrap();
            let mut stack: [$scalar; 2] = [sentinel; 2];
            gradient.read_into(&mut stack).unwrap();
            assert_eq!(
                stack.map(<$scalar>::to_bits),
                [zero.to_bits(), sentinel.to_bits()]
            );
            fault(
                input_gradient::<$scalar>(&[0.0], &weights, &[1.0]),
                PcuExecutionFaultKind::ArithmeticOverflow,
            );
            fault(
                forward_unused_fault::<$scalar>(&[0.0], &weights, &[target]),
                PcuExecutionFaultKind::DivideByZero,
            );
            weights.read_into(&mut stack).unwrap();
            assert_eq!(stack[0].to_bits(), <$scalar>::MAX.to_bits());
            global::clear_thread_cache().unwrap();
            gradient.read_into(&mut stack).unwrap();
            assert_eq!(stack[0].to_bits(), zero.to_bits());
            drop(gradient);
        }
        let retry = weight_gradient::<$scalar>(&[0.0], &weights, &[1.0]).unwrap();
        drop(weights);
        global::clear_thread_cache().unwrap();
        let mut stack: [$scalar; 2] = [sentinel; 2];
        retry.read_into(&mut stack).unwrap();
        assert_eq!(
            stack.map(<$scalar>::to_bits),
            [zero.to_bits(), sentinel.to_bits()]
        );
    }};
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend,
        ..Default::default()
    })
    .unwrap();
    verify_format!(f32);
    verify_format!(f64);
    global::use_defaults().unwrap();
}

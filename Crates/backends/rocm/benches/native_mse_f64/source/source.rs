//! Actual authored owned scalar loss under explicit, independent native permissions.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
    PcuScalar,
};

#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(preserve_precision),
    flag(non_deterministic)
)]
pub fn preserve<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(backend_precision),
    flag(non_deterministic)
)]
pub fn optimized<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu]
pub fn checked<T: PcuScalar>(
    prediction: &[T; 1],
    target: &[T; 1],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(strict), flag(native_compound), flag(non_deterministic))]
pub fn strict<T: PcuScalar>(
    prediction: &[T; 1],
    target: &[T; 1],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(non_strict), flag(native_compound), flag(deterministic))]
pub fn portable<T: PcuScalar>(
    prediction: &[T; 1],
    target: &[T; 1],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(non_deterministic),
    flag(reject_subnormal_result)
)]
pub fn tight<T: PcuScalar>(
    prediction: &[T; 1],
    target: &[T; 1],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

pub fn loss<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
    optimized_precision: bool,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    if optimized_precision {
        optimized(prediction, target)
    } else {
        preserve(prediction, target)
    }
}

#[pcu]
pub fn identity<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

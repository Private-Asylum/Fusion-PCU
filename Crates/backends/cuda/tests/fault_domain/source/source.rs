//! Genuine ordered compound source; later events exceed output/launch lane counts.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(crate_path=::pcu_facade,flag(strict))]
pub fn product<T: PcuScalar>(
    a: &[[T; 3]; 1],
    b: &[[T; 1]; 3],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(a, b)
}
#[pcu(crate_path=::pcu_facade,flag(strict))]
pub fn update<T: PcuScalar>(a: &[T; 3], b: &[T; 3]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(a, b, 1.0)
}
#[pcu(crate_path=::pcu_facade,flag(strict))]
pub fn loss<T: PcuScalar>(a: &[T; 3], b: &[T; 3]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(a, b)
}

#[pcu(crate_path=::pcu_facade, invocations=4)]
pub fn composed<T: pcu_facade::PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = (left[id] + right[id]) * left[id];
}

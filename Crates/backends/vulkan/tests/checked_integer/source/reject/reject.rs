//! Actual generic annotated workloads preserve the selected scalar format.
use pcu_facade::pcu;
use pcu_facade::PcuCheckedInteger;
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn add<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn sub<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn mul<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * right[id];
}

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(strict))]
pub fn grid<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn scale<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * *right;
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict))]
pub fn strict<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}

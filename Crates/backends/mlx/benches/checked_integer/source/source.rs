//! Genuine independently prepared integer source peers.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedInteger,
};
#[pcu(invocations=N,flag(strict),crate_path=::pcu_facade)]
pub fn add<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(strict),flag(clamp_range),crate_path=::pcu_facade)]
pub fn add_clamp<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(strict),crate_path=::pcu_facade)]
pub fn sub<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,flag(strict),flag(clamp_range),crate_path=::pcu_facade)]
pub fn sub_clamp<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,flag(strict),crate_path=::pcu_facade)]
pub fn mul<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,flag(strict),flag(clamp_range),crate_path=::pcu_facade)]
pub fn mul_clamp<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}

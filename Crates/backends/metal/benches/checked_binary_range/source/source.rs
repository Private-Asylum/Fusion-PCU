//! Genuine six-format checked binary source; policy/type specialization remains cold.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuCheckedFloat};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn add_ieee_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn add_ieee_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn add_gradual_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),flag(clamp_range),crate_path=::pcu_facade)]
pub fn add_gradual_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(strict),flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn add_tight_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(strict),flag(reject_subnormal_result),flag(clamp_range),crate_path=::pcu_facade)]
pub fn add_tight_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn sub_ieee_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn sub_ieee_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn sub_gradual_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),flag(clamp_range),crate_path=::pcu_facade)]
pub fn sub_gradual_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,flag(strict),flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn sub_tight_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,flag(strict),flag(reject_subnormal_result),flag(clamp_range),crate_path=::pcu_facade)]
pub fn sub_tight_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn mul_ieee_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn mul_ieee_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn mul_gradual_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),flag(clamp_range),crate_path=::pcu_facade)]
pub fn mul_gradual_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,flag(strict),flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn mul_tight_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,flag(strict),flag(reject_subnormal_result),flag(clamp_range),crate_path=::pcu_facade)]
pub fn mul_tight_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn div_ieee_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn div_ieee_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn div_gradual_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),flag(clamp_range),crate_path=::pcu_facade)]
pub fn div_gradual_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations=N,flag(strict),flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn div_tight_reject<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations=N,flag(strict),flag(reject_subnormal_result),flag(clamp_range),crate_path=::pcu_facade)]
pub fn div_tight_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}

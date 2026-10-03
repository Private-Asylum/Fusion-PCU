//! Genuine source declarations retain an unread empty slice and repeated indexed input.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuCheckedInteger,
};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn float_add_ieee_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_add_ieee_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn float_add_gradual_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_add_gradual_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn float_add_tight_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result),flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_add_tight_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn float_sub_ieee_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_sub_ieee_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn float_sub_gradual_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_sub_gradual_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn float_sub_tight_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result),flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_sub_tight_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn float_mul_ieee_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_mul_ieee_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn float_mul_gradual_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_mul_gradual_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn float_mul_tight_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result),flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_mul_tight_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn float_div_ieee_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] / input[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_div_ieee_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] / input[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn float_div_gradual_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] / input[id];
}
#[pcu(invocations=N,flag(allow_gradual_underflow),flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_div_gradual_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] / input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn float_div_tight_reject<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] / input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result),flag(clamp_range),crate_path=::pcu_facade)]
pub fn float_div_tight_clamp<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] / input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn integer_add_integer_reject<T: PcuCheckedInteger, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn integer_add_integer_clamp<T: PcuCheckedInteger, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn integer_sub_integer_reject<T: PcuCheckedInteger, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn integer_sub_integer_clamp<T: PcuCheckedInteger, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn integer_mul_integer_reject<T: PcuCheckedInteger, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn integer_mul_integer_clamp<T: PcuCheckedInteger, const N: usize>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}

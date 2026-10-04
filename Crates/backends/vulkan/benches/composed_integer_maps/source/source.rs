//! Genuine two-effect composed integer source; reproducibility remains independently requested.
#[rustfmt::skip]
use pcu_facade::{pcu, PcuCheckedInteger};
#[pcu(invocations = 65, crate_path = ::pcu_facade)]
pub fn normal<T: PcuCheckedInteger>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
#[pcu(invocations = 65, flag(deterministic), crate_path = ::pcu_facade)]
pub fn portable<T: PcuCheckedInteger>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}

#[pcu(invocations = 65, crate_path = ::pcu_facade)]
pub fn dead_normal<T: PcuCheckedInteger>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    let _discarded: T = input[id] + input[id];
    output[id] = input[id];
}
#[pcu(invocations = 65, flag(deterministic), crate_path = ::pcu_facade)]
pub fn dead_portable<T: PcuCheckedInteger>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    let _discarded: T = input[id] + input[id];
    output[id] = input[id];
}

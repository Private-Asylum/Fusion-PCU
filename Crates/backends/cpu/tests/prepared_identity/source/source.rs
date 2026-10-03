//! Generic source preserves representation without granting arithmetic support.
#[rustfmt::skip]
use pcu_facade::{pcu, PcuScalar};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn copy<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}

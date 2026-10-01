//! Shared real annotated source for integration tests and Criterion peers.

use pcu_facade::pcu;

#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn negate<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

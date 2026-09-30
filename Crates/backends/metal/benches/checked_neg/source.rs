//! Actual source authoring boundary used by the canonical Metal benchmark.
use pcu_facade::pcu;

#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn negate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

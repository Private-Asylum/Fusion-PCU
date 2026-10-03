//! The same ordinary annotated source supplies global and explicit prepared calls.
use fusion_pcu::pcu;

#[pcu(invocations = N, flag(strict))]
pub fn negate<const N: usize>(input: &[f32; N], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = N, flag(strict))]
pub fn add<const N: usize>(lhs: &[u64; N], rhs: &[u64; N], output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] + rhs[id];
}
#[pcu(invocations = N, flag(strict))]
pub fn mul<const N: usize>(lhs: &[u64; N], rhs: &[u64; N], output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] * rhs[id];
}

#[pcu(invocations = N, flag(strict))]
pub fn composed_locals<const N: usize>(input: &[f32; N], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    let value = input[id];
    let doubled = value + value;
    output[id] = doubled * value;
}

#[pcu]
fn helper_locals(value: f32) -> f32 {
    let mut value: f32 = value;
    let saved = value;
    value += saved;
    value *= saved;
    value
}

#[pcu(invocations = N, flag(strict))]
pub fn composed_helper<const N: usize>(input: &[f32; N], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = helper_locals(input[id]);
}

#[pcu]
#[allow(clippy::missing_const_for_fn)] // PCU scalar companions have a non-const lowering contract.
fn integer_helper(mut value: u64, seed: u64) -> u64 {
    let original = value;
    value += seed;
    value *= original;
    value - original
}

#[pcu(invocations = N, flag(strict))]
pub fn composed_integer_helper<const N: usize>(input: &[u64; N], seed: &u64, output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = integer_helper(input[id], *seed);
}

#[pcu(invocations = N, flag(strict))]
pub fn ordered_stores<const N: usize>(input: &[f32; N], stage: &mut [f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    stage[id] = original + original;
    let updated = stage[id];
    output[id] = updated * original;
}

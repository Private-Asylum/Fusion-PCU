//! Genuine generic checked unary source; specialization and policy selection are cold.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};
#[pcu(flag(deterministic),invocations=N,crate_path=::pcu_facade)]
pub fn negate<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(flag(deterministic),invocations=N,crate_path=::pcu_facade)]
pub fn relu<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[pcu(flag(deterministic),invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn negate_gradual<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(flag(deterministic),invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn relu_gradual<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[pcu(flag(deterministic),invocations=N,flag(strict),flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn negate_tight<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(flag(deterministic),invocations=N,flag(strict),flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn relu_tight<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(flag(deterministic),flag(clamp_range),invocations=N,crate_path=::pcu_facade)]
pub fn negate_clamp<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(flag(deterministic),flag(clamp_range),invocations=N,crate_path=::pcu_facade)]
pub fn relu_clamp<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(flag(deterministic),flag(clamp_range),invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn negate_gradual_clamp<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(flag(deterministic),flag(clamp_range),invocations=N,flag(allow_gradual_underflow),crate_path=::pcu_facade)]
pub fn relu_gradual_clamp<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(flag(deterministic),flag(clamp_range),invocations=N,flag(strict),flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn negate_tight_clamp<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(flag(deterministic),flag(clamp_range),invocations=N,flag(strict),flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn relu_tight_clamp<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

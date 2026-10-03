//! Genuine checked low-format unary maps with explicit range/underflow requirements.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuCheckedFloat};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn neg_reject_ieee<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(allow_gradual_underflow))]
pub fn neg_reject_gradual<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(strict),flag(reject_subnormal_result))]
pub fn neg_reject_strict<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn neg_clamp_ieee<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range),flag(allow_gradual_underflow))]
pub fn neg_clamp_gradual<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range),flag(strict),flag(reject_subnormal_result))]
pub fn neg_clamp_strict<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn relu_reject_ieee<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(allow_gradual_underflow))]
pub fn relu_reject_gradual<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(strict),flag(reject_subnormal_result))]
pub fn relu_reject_strict<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn relu_clamp_ieee<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range),flag(allow_gradual_underflow))]
pub fn relu_clamp_gradual<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range),flag(strict),flag(reject_subnormal_result))]
pub fn relu_clamp_strict<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations=3,crate_path=::pcu_facade,flag(clamp_range),flag(reject_subnormal_result))]
pub fn grid_neg<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range),flag(reject_subnormal_result))]
pub fn broadcast_relu<T: PcuCheckedFloat, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(*input);
}

#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn neg_broadcast_reject_ieee<T: PcuCheckedFloat, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -*input;
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(allow_gradual_underflow))]
pub fn neg_broadcast_reject_gradual<T: PcuCheckedFloat, const N: usize>(
    input: &T,
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    output[id] = -*input;
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(strict),flag(reject_subnormal_result))]
pub fn neg_broadcast_reject_strict<T: PcuCheckedFloat, const N: usize>(
    input: &T,
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    output[id] = -*input;
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn neg_broadcast_clamp_ieee<T: PcuCheckedFloat, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -*input;
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range),flag(allow_gradual_underflow))]
pub fn neg_broadcast_clamp_gradual<T: PcuCheckedFloat, const N: usize>(
    input: &T,
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    output[id] = -*input;
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range),flag(strict),flag(reject_subnormal_result))]
pub fn neg_broadcast_clamp_strict<T: PcuCheckedFloat, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -*input;
}

#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn relu_broadcast_reject_ieee<T: PcuCheckedFloat, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(*input);
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(allow_gradual_underflow))]
pub fn relu_broadcast_reject_gradual<T: PcuCheckedFloat, const N: usize>(
    input: &T,
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(*input);
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(strict),flag(reject_subnormal_result))]
pub fn relu_broadcast_reject_strict<T: PcuCheckedFloat, const N: usize>(
    input: &T,
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(*input);
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn relu_broadcast_clamp_ieee<T: PcuCheckedFloat, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(*input);
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range),flag(allow_gradual_underflow))]
pub fn relu_broadcast_clamp_gradual<T: PcuCheckedFloat, const N: usize>(
    input: &T,
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(*input);
}

#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range),flag(strict),flag(reject_subnormal_result))]
pub fn relu_broadcast_clamp_strict<T: PcuCheckedFloat, const N: usize>(
    input: &T,
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(*input);
}

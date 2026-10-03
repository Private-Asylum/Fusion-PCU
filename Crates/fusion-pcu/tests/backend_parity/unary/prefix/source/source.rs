//! Genuine annotated unary source entries; local Clamp stays separate from owner creation.
use super::pcu;
use super::PcuCheckedFloat;

#[pcu(invocations = 7)]
pub(super) fn direct_neg<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = 7)]
pub(super) fn direct_relu<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 1)]
pub(super) fn grid_neg<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = -input[id];
        id += stride;
    }
}

#[pcu(invocations = 1)]
pub(super) fn grid_relu<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}

#[pcu(invocations = 7)]
pub(super) fn broadcast_neg<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(invocations = 7)]
pub(super) fn broadcast_relu<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

#[pcu(invocations = 7, flag(clamp_range))]
pub(super) fn clamp_direct_neg<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = 7, flag(clamp_range))]
pub(super) fn clamp_direct_relu<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 1, flag(clamp_range))]
pub(super) fn clamp_grid_neg<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = -input[id];
        id += stride;
    }
}

#[pcu(invocations = 1, flag(clamp_range))]
pub(super) fn clamp_grid_relu<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}

#[pcu(invocations = 7, flag(clamp_range))]
pub(super) fn clamp_broadcast_neg<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(invocations = 7, flag(clamp_range))]
pub(super) fn clamp_broadcast_relu<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

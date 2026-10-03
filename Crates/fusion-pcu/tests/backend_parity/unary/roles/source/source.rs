//! Ordinary unary functions keep declaration order separate from actual resources.
use super::pcu;
use super::PcuCheckedFloat;

#[pcu(invocations = 7)]
pub(super) fn direct_neg<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = 7)]
pub(super) fn direct_relu<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 3)]
pub(super) fn grid_neg<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = -input[id];
        id += stride;
    }
}

#[pcu(invocations = 3)]
pub(super) fn grid_relu<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}

#[pcu(invocations = 7)]
pub(super) fn broadcast_neg<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(invocations = 7)]
pub(super) fn broadcast_relu<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

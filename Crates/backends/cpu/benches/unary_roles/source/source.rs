//! Genuine source specializations freeze operation policies before detached preparation.
use super::pcu;
use super::PcuCheckedFloat;
#[pcu(invocations = 7, crate_path = ::pcu_facade)]
pub(super) fn direct_neg_reject_ieee<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade)]
pub(super) fn direct_relu_reject_ieee<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 3, crate_path = ::pcu_facade)]
pub(super) fn grid_neg_reject_ieee<T: PcuCheckedFloat>(
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

#[pcu(invocations = 3, crate_path = ::pcu_facade)]
pub(super) fn grid_relu_reject_ieee<T: PcuCheckedFloat>(
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

#[pcu(invocations = 7, crate_path = ::pcu_facade)]
pub(super) fn broadcast_neg_reject_ieee<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade)]
pub(super) fn broadcast_relu_reject_ieee<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub(super) fn direct_neg_reject_gradual<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub(super) fn direct_relu_reject_gradual<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub(super) fn grid_neg_reject_gradual<T: PcuCheckedFloat>(
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

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub(super) fn grid_relu_reject_gradual<T: PcuCheckedFloat>(
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

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub(super) fn broadcast_neg_reject_gradual<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub(super) fn broadcast_relu_reject_gradual<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(reject_subnormal_result))]
pub(super) fn direct_neg_reject_tight<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(reject_subnormal_result))]
pub(super) fn direct_relu_reject_tight<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(reject_subnormal_result))]
pub(super) fn grid_neg_reject_tight<T: PcuCheckedFloat>(
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

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(reject_subnormal_result))]
pub(super) fn grid_relu_reject_tight<T: PcuCheckedFloat>(
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

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(reject_subnormal_result))]
pub(super) fn broadcast_neg_reject_tight<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(reject_subnormal_result))]
pub(super) fn broadcast_relu_reject_tight<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(clamp_range))]
pub(super) fn direct_neg_clamp_ieee<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(clamp_range))]
pub(super) fn direct_relu_clamp_ieee<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(clamp_range))]
pub(super) fn grid_neg_clamp_ieee<T: PcuCheckedFloat>(
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

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(clamp_range))]
pub(super) fn grid_relu_clamp_ieee<T: PcuCheckedFloat>(
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

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(clamp_range))]
pub(super) fn broadcast_neg_clamp_ieee<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(clamp_range))]
pub(super) fn broadcast_relu_clamp_ieee<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub(super) fn direct_neg_clamp_gradual<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub(super) fn direct_relu_clamp_gradual<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub(super) fn grid_neg_clamp_gradual<T: PcuCheckedFloat>(
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

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub(super) fn grid_relu_clamp_gradual<T: PcuCheckedFloat>(
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

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub(super) fn broadcast_neg_clamp_gradual<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub(super) fn broadcast_relu_clamp_gradual<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(reject_subnormal_result), flag(clamp_range))]
pub(super) fn direct_neg_clamp_tight<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(reject_subnormal_result), flag(clamp_range))]
pub(super) fn direct_relu_clamp_tight<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(reject_subnormal_result), flag(clamp_range))]
pub(super) fn grid_neg_clamp_tight<T: PcuCheckedFloat>(
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

#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(reject_subnormal_result), flag(clamp_range))]
pub(super) fn grid_relu_clamp_tight<T: PcuCheckedFloat>(
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

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(reject_subnormal_result), flag(clamp_range))]
pub(super) fn broadcast_neg_clamp_tight<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(invocations = 7, crate_path = ::pcu_facade, flag(reject_subnormal_result), flag(clamp_range))]
pub(super) fn broadcast_relu_clamp_tight<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

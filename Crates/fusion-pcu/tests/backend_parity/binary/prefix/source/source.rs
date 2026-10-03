//! Actual annotated source peers for automatic full-capacity binary borrowing.
use super::pcu;
use super::PcuCheckedFloat;

macro_rules! kernel {
    ($name:ident, $clamp:ident, $op:tt, $left:tt, $right:tt) => {
        #[pcu(invocations = 7)]
        pub(super) fn $name<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
            let id = pcu::context::global_invocation_id();
            output[id] = left[$left] $op right[$right];
        }
        #[pcu(invocations = 7, flag(clamp_range))]
        pub(super) fn $clamp<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
            let id = pcu::context::global_invocation_id();
            output[id] = left[$left] $op right[$right];
        }
    };
}

kernel!(add, clamp_add, +, id, id);
kernel!(sub, clamp_sub, -, id, id);
kernel!(mul, clamp_mul, *, id, id);
kernel!(div, clamp_div, /, id, id);
kernel!(left_broadcast, clamp_left_broadcast, +, 0, id);
kernel!(right_broadcast, clamp_right_broadcast, +, id, 0);

#[pcu(invocations = 1)]
pub(super) fn grid<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = left[id] + right[id];
        id += stride;
    }
}

#[pcu(invocations = 1, flag(clamp_range))]
pub(super) fn clamp_grid<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = left[id] + right[id];
        id += stride;
    }
}

#[pcu(invocations = 7)]
pub(super) fn repeated<T: PcuCheckedFloat>(left: &[T], unused: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + left[id];
}

#[pcu(invocations = 7, flag(clamp_range))]
pub(super) fn clamp_repeated<T: PcuCheckedFloat>(left: &[T], unused: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + left[id];
}

//! Genuine integer source roles and independent range policies specialize cold.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedInteger,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict))]
pub fn add_reject_0<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(clamp_range))]
pub fn add_clamp_0<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict))]
pub fn sub_reject_0<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(clamp_range))]
pub fn sub_clamp_0<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict))]
pub fn mul_reject_0<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(clamp_range))]
pub fn mul_clamp_0<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict))]
pub fn add_reject_1<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(clamp_range))]
pub fn add_clamp_1<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict))]
pub fn sub_reject_1<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] - right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(clamp_range))]
pub fn sub_clamp_1<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] - right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict))]
pub fn mul_reject_1<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(clamp_range))]
pub fn mul_clamp_1<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict))]
pub fn add_reject_2<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] + left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(clamp_range))]
pub fn add_clamp_2<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] + left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict))]
pub fn sub_reject_2<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] - left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(clamp_range))]
pub fn sub_clamp_2<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] - left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict))]
pub fn mul_reject_2<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] * left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(clamp_range))]
pub fn mul_clamp_2<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] * left[id];
}

#[pcu(crate_path = ::pcu_facade)]
pub fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

//! Genuine generic checked low-precision binary maps; all flags remain independent.
use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::{ PcuCheckedFloat, PcuScalar, PcuTensor, PcuExecutionError };
pub mod add {
    use super::{pcu, PcuCheckedFloat};
    #[pcu(invocations=N, flag(deterministic))]
    pub fn portable<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + right[id];
    }
    #[pcu(invocations=N, flag(deterministic), flag(strict))]
    pub fn portable_strict<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + right[id];
    }
    #[pcu(invocations = 3, flag(deterministic))]
    pub fn portable_grid<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = 3, flag(deterministic), flag(allow_gradual_underflow))]
    pub fn portable_allow<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = 3, flag(deterministic), flag(reject_subnormal_result))]
    pub fn portable_tight<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations=N, flag(deterministic))]
    pub fn portable_broadcast<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &T,
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + *right;
    }
    #[pcu(invocations = N)]
    pub fn direct<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + right[id];
    }
    #[pcu(invocations = N, flag(strict))]
    pub fn strict<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + right[id];
    }
    #[pcu(invocations = 3)]
    pub fn grid<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] + right[id];
            id += stride;
        }
    }
    #[pcu(invocations = N, flag(allow_gradual_underflow))]
    pub fn allow<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + right[id];
    }
    #[pcu(invocations = N, flag(reject_subnormal_result))]
    pub fn tight<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + right[id];
    }
    #[pcu(invocations = 3, flag(clamp_range))]
    pub fn clamp<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] + right[id];
            id += stride;
        }
    }
    #[pcu(invocations = 3, flag(clamp_range), flag(allow_gradual_underflow))]
    pub fn clamp_allow<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = 3, flag(clamp_range), flag(reject_subnormal_result))]
    pub fn clamp_tight<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = N)]
    pub fn broadcast<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &T, output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + *right;
    }
}
pub mod sub {
    use super::{pcu, PcuCheckedFloat};
    #[pcu(invocations=N, flag(deterministic))]
    pub fn portable<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - right[id];
    }
    #[pcu(invocations=N, flag(deterministic), flag(strict))]
    pub fn portable_strict<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - right[id];
    }
    #[pcu(invocations = 3, flag(deterministic))]
    pub fn portable_grid<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = 3, flag(deterministic), flag(allow_gradual_underflow))]
    pub fn portable_allow<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = 3, flag(deterministic), flag(reject_subnormal_result))]
    pub fn portable_tight<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations=N, flag(deterministic))]
    pub fn portable_broadcast<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &T,
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - *right;
    }
    #[pcu(invocations = N)]
    pub fn direct<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - right[id];
    }
    #[pcu(invocations = N, flag(strict))]
    pub fn strict<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - right[id];
    }
    #[pcu(invocations = 3)]
    pub fn grid<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] - right[id];
            id += stride;
        }
    }
    #[pcu(invocations = N, flag(allow_gradual_underflow))]
    pub fn allow<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - right[id];
    }
    #[pcu(invocations = N, flag(reject_subnormal_result))]
    pub fn tight<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - right[id];
    }
    #[pcu(invocations = 3, flag(clamp_range))]
    pub fn clamp<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] - right[id];
            id += stride;
        }
    }
    #[pcu(invocations = 3, flag(clamp_range), flag(allow_gradual_underflow))]
    pub fn clamp_allow<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = 3, flag(clamp_range), flag(reject_subnormal_result))]
    pub fn clamp_tight<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = N)]
    pub fn broadcast<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &T, output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - *right;
    }
}
pub mod mul {
    use super::{pcu, PcuCheckedFloat};
    #[pcu(invocations=N, flag(deterministic))]
    pub fn portable<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * right[id];
    }
    #[pcu(invocations=N, flag(deterministic), flag(strict))]
    pub fn portable_strict<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * right[id];
    }
    #[pcu(invocations = 3, flag(deterministic))]
    pub fn portable_grid<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = 3, flag(deterministic), flag(allow_gradual_underflow))]
    pub fn portable_allow<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = 3, flag(deterministic), flag(reject_subnormal_result))]
    pub fn portable_tight<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations=N, flag(deterministic))]
    pub fn portable_broadcast<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &T,
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * *right;
    }
    #[pcu(invocations = N)]
    pub fn direct<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * right[id];
    }
    #[pcu(invocations = N, flag(strict))]
    pub fn strict<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * right[id];
    }
    #[pcu(invocations = 3)]
    pub fn grid<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] * right[id];
            id += stride;
        }
    }
    #[pcu(invocations = N, flag(allow_gradual_underflow))]
    pub fn allow<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * right[id];
    }
    #[pcu(invocations = N, flag(reject_subnormal_result))]
    pub fn tight<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * right[id];
    }
    #[pcu(invocations = 3, flag(clamp_range))]
    pub fn clamp<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] * right[id];
            id += stride;
        }
    }
    #[pcu(invocations = 3, flag(clamp_range), flag(allow_gradual_underflow))]
    pub fn clamp_allow<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = 3, flag(clamp_range), flag(reject_subnormal_result))]
    pub fn clamp_tight<T: PcuCheckedFloat, const N: usize>(
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
    #[pcu(invocations = N)]
    pub fn broadcast<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &T, output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * *right;
    }
}
pub mod div {
    use super::{pcu, PcuCheckedFloat};
    #[pcu(invocations=N, flag(deterministic))]
    pub fn portable<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] / right[id];
    }
    #[pcu(invocations=N, flag(deterministic), flag(strict))]
    pub fn portable_strict<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] / right[id];
    }
    #[pcu(invocations = 3, flag(deterministic))]
    pub fn portable_grid<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] / right[id];
            id += stride;
        }
    }
    #[pcu(invocations = 3, flag(deterministic), flag(allow_gradual_underflow))]
    pub fn portable_allow<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] / right[id];
            id += stride;
        }
    }
    #[pcu(invocations = 3, flag(deterministic), flag(reject_subnormal_result))]
    pub fn portable_tight<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] / right[id];
            id += stride;
        }
    }
    #[pcu(invocations=N, flag(deterministic))]
    pub fn portable_broadcast<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &T,
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] / *right;
    }
    #[pcu(invocations = N)]
    pub fn direct<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] / right[id];
    }
    #[pcu(invocations = N, flag(strict))]
    pub fn strict<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] / right[id];
    }
    #[pcu(invocations = 3)]
    pub fn grid<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] / right[id];
            id += stride;
        }
    }
    #[pcu(invocations = N, flag(allow_gradual_underflow))]
    pub fn allow<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] / right[id];
    }
    #[pcu(invocations = N, flag(reject_subnormal_result))]
    pub fn tight<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] / right[id];
    }
    #[pcu(invocations = 3, flag(clamp_range))]
    pub fn clamp<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] / right[id];
            id += stride;
        }
    }
    #[pcu(invocations = 3, flag(clamp_range), flag(allow_gradual_underflow))]
    pub fn clamp_allow<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] / right[id];
            id += stride;
        }
    }
    #[pcu(invocations = 3, flag(clamp_range), flag(reject_subnormal_result))]
    pub fn clamp_tight<T: PcuCheckedFloat, const N: usize>(
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] / right[id];
            id += stride;
        }
    }
    #[pcu(invocations = N)]
    pub fn broadcast<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &T, output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] / *right;
    }
}
#[pcu]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(invocations = N)]
pub fn unsupported_compound<T: PcuCheckedFloat, const N: usize>(
    a: &[T],
    b: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -(a[id] + b[id]);
}

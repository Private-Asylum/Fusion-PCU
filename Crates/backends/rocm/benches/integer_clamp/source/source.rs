//! Genuine generic checked wide integer source maps; explicit recovered Clamp range.
#[rustfmt::skip]
use fusion_pcu::{pcu,PcuCheckedInteger,PcuScalar,PcuTensor,PcuExecutionError};
#[pcu]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
pub mod add {
    use super::{pcu, PcuCheckedInteger};
    #[pcu(invocations=N,flag(clamp_range))]
    pub fn direct<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + right[id];
    }
    #[pcu(invocations = 3, flag(clamp_range))]
    pub fn grid<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] + right[id];
            id += stride;
        }
    }
    #[pcu(invocations=N,flag(clamp_range))]
    pub fn broadcast<T: PcuCheckedInteger, const N: usize>(
        left: &[T],
        right: &T,
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] + right;
    }
}
pub mod sub {
    use super::{pcu, PcuCheckedInteger};
    #[pcu(invocations=N,flag(clamp_range))]
    pub fn direct<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - right[id];
    }
    #[pcu(invocations = 3, flag(clamp_range))]
    pub fn grid<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] - right[id];
            id += stride;
        }
    }
    #[pcu(invocations=N,flag(clamp_range))]
    pub fn broadcast<T: PcuCheckedInteger, const N: usize>(
        left: &[T],
        right: &T,
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] - right;
    }
}
pub mod mul {
    use super::{pcu, PcuCheckedInteger};
    #[pcu(invocations=N,flag(clamp_range))]
    pub fn direct<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * right[id];
    }
    #[pcu(invocations = 3, flag(clamp_range))]
    pub fn grid<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = left[id] * right[id];
            id += stride;
        }
    }
    #[pcu(invocations=N,flag(clamp_range))]
    pub fn broadcast<T: PcuCheckedInteger, const N: usize>(
        left: &[T],
        right: &T,
        output: &mut [T],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = left[id] * right;
    }
}

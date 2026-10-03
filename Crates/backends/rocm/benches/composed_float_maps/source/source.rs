//! Genuine composed checked maps; source, prepared IR and native use the same chain.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
pub mod map {
    #[rustfmt::skip]
    use super::{
        pcu,
        PcuCheckedFloat,
    };
    #[pcu(crate_path=::pcu_facade,invocations=N)]
    pub fn single<T: PcuCheckedFloat, const N: usize>(output: &mut [T], left: &[T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = (left[id] + left[id]) * left[id];
    }
    #[pcu(crate_path=::pcu_facade,invocations=N)]
    pub fn unused<T: PcuCheckedFloat, const N: usize>(unused: &[T], output: &mut [T], left: &[T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = (left[id] + left[id]) * left[id];
    }
    #[pcu(crate_path=::pcu_facade,invocations=17)]
    pub fn grid<T: PcuCheckedFloat, const N: usize>(unused: &[T], output: &mut [T], left: &[T]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            output[id] = (left[id] + left[id]) * left[id];
            id += stride;
        }
    }
    #[pcu(crate_path=::pcu_facade,invocations=N)]
    pub fn seed<T: PcuCheckedFloat, const N: usize>(unused: &[T], output: &mut [T], seed: &T) {
        let id = pcu::context::global_invocation_id();
        output[id] = (*seed + *seed) * *seed;
    }
}

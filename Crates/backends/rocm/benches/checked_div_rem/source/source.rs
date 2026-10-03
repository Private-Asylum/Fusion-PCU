//! Genuine annotated checked quotient/remainder profiles; strictness does not grant portability.
use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
macro_rules! profiles {
    ($($module:ident: $ty:ty),* $(,)?) => {$(
        pub mod $module {
            use super::pcu;
            #[pcu(invocations = N)]
            pub fn direct<const N: usize>(lhs: &[$ty], rhs: &[$ty], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
                quotient[id] = q;
                remainder[id] = r;
            }
            #[pcu(invocations = N, flag(strict))]
            pub fn strict<const N: usize>(lhs: &[$ty], rhs: &[$ty], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
                quotient[id] = q;
                remainder[id] = r;
            }
            #[pcu(invocations = 3)]
            pub fn grid<const N: usize>(lhs: &[$ty], rhs: &[$ty], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let mut id = pcu::context::global_invocation_id();
                let stride = pcu::context::invocation_count();
                while id < N {
                    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
                    quotient[id] = q;
                    remainder[id] = r;
                    id += stride;
                }
            }
        }
    )*};
}
profiles!(signed8: i8, unsigned8: u8, signed16: i16, unsigned16: u16,
    signed32: i32, unsigned32: u32, signed64: i64, unsigned64: u64);
#[pcu]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

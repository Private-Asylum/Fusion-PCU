//! Genuine direct and grid annotated quotient/remainder entries for all eight widths.
macro_rules! profiles {
    ($($module:ident: $ty:ty),* $(,)?) => {$(
        pub mod $module {
            use pcu_facade::pcu;
            #[pcu(invocations = N, crate_path = ::pcu_facade)]
            pub fn direct<const N: usize>(lhs: &[$ty], rhs: &[$ty], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let id = context.global_invocation_id;
                let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
                quotient[id] = q;
                remainder[id] = r;
            }
            #[pcu(invocations = N, flag(strict), crate_path = ::pcu_facade)]
            pub fn strict<const N: usize>(lhs: &[$ty], rhs: &[$ty], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let id = context.global_invocation_id;
                let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
                quotient[id] = q;
                remainder[id] = r;
            }
            #[pcu(invocations = 3, crate_path = ::pcu_facade)]
            pub fn grid<const N: usize>(lhs: &[$ty], rhs: &[$ty], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let mut id = context.global_invocation_id;
                let stride = context.invocation_count;
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
profiles!(i8_source: i8, u8_source: u8, i16_source: i16, u16_source: u16,
    i32_source: i32, u32_source: u32, i64_source: i64, u64_source: u64);

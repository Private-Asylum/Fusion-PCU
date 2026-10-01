//! Twenty-four ordinary annotated width/operation entries shared by tests and Criterion.

macro_rules! integer_source {
    ($name:ident, $ty:ty) => {
        pub mod $name {
            use pcu_facade::pcu;

            #[pcu(invocations = N, crate_path = ::pcu_facade)]
            pub fn add<const N: usize>(lhs: &[$ty], rhs: &[$ty], output: &mut [$ty]) {
                let id = context.global_invocation_id;
                output[id] = lhs[id] + rhs[id];
            }

            #[pcu(invocations = N, crate_path = ::pcu_facade)]
            pub fn sub<const N: usize>(lhs: &[$ty], rhs: &[$ty], output: &mut [$ty]) {
                let id = context.global_invocation_id;
                output[id] = lhs[id] - rhs[id];
            }

            #[pcu(invocations = N, crate_path = ::pcu_facade)]
            pub fn mul<const N: usize>(lhs: &[$ty], rhs: &[$ty], output: &mut [$ty]) {
                let id = context.global_invocation_id;
                output[id] = lhs[id] * rhs[id];
            }
        }
    };
}

integer_source!(i8_source, i8);
integer_source!(u8_source, u8);
integer_source!(i16_source, i16);
integer_source!(u16_source, u16);
integer_source!(i32_source, i32);
integer_source!(u32_source, u32);
integer_source!(i64_source, i64);
integer_source!(u64_source, u64);

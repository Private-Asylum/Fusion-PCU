//! Eight actual annotated checked source workloads, reused by the native benchmark pairs.
macro_rules! binary_source {
    ($module:ident, $ty:ty) => {
        pub mod $module {
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
            #[pcu(invocations = N, crate_path = ::pcu_facade)]
            pub fn div<const N: usize>(lhs: &[$ty], rhs: &[$ty], output: &mut [$ty]) {
                let id = context.global_invocation_id;
                output[id] = lhs[id] / rhs[id];
            }
        }
    };
}
binary_source!(f32_source, f32);
binary_source!(f64_source, f64);

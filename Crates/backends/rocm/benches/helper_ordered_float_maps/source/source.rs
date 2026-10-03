//! Concrete scalar helpers preserve saved aliases and lower compound updates into checked SSA.
use pcu_facade::pcu;
macro_rules! family {
    ($module:ident, $ty:ty) => {
        pub mod $module {
            use super::pcu;
            #[pcu(crate_path=::pcu_facade)]
            pub fn doubled(mut value: $ty) -> $ty {
                let original: $ty = value;
                value += original;
                value
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn product(mut value: $ty, original: $ty) -> $ty {
                value *= original;
                value
            }
            #[pcu(crate_path=::pcu_facade,invocations=N)]
            pub fn direct<const N: usize>(stage: &mut [$ty], output: &mut [$ty], input: &[$ty]) {
                let id = pcu::context::global_invocation_id();
                let original: $ty = input[id];
                stage[id] = doubled(original);
                let reloaded: $ty = stage[id];
                output[id] = product(reloaded, original);
            }
            #[pcu(crate_path=::pcu_facade,invocations=17)]
            pub fn grid<const N: usize>(stage: &mut [$ty], output: &mut [$ty], input: &[$ty]) {
                let mut id = pcu::context::global_invocation_id();
                let stride = pcu::context::invocation_count();
                while id < N {
                    let original: $ty = input[id];
                    stage[id] = doubled(original);
                    let reloaded: $ty = stage[id];
                    output[id] = product(reloaded, original);
                    id += stride;
                }
            }
        }
    };
}
family!(single, f32);
family!(double, f64);

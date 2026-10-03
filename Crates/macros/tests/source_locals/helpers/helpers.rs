//! A helper and its separately authored inline source retain the same checked SSA.
use fusion_pcu_macros::pcu;

#[path = "integer/integer.rs"]
mod integer;

#[pcu(crate_path = ::pcu_alias)]
#[allow(clippy::let_with_type_underscore)] // Exercise the admitted inferred annotation spelling.
fn shifted(mut value: f32, seed: f32) -> f32 {
    let original: _ = value;
    value += seed;
    let shifted = value * original;
    shifted + original
}

mod nested {
    use fusion_pcu_macros::pcu;
    #[pcu(crate_path = ::pcu_alias)]
    pub fn shifted(mut value: f64, seed: f64) -> f64 {
        let original = value;
        value += seed;
        let scaled = value * original;
        scaled + original
    }

    #[pcu(crate_path = ::pcu_alias)]
    pub fn forward(value: f64, seed: f64) -> f64 {
        let saved: f64 = shifted(value, seed);
        saved
    }
}
use nested::forward as helper_alias;

macro_rules! sources {
    ($ty:ident, $helper:ident, $direct:ident, $flat:ident, $grid:ident, $flat_grid:ident) => {
        #[pcu(invocations = N, crate_path = ::pcu_alias)]
        fn $direct<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
            let id = pcu::context::global_invocation_id();
            output[id] = $helper(input[id], *seed);
        }
        #[pcu(invocations = N, crate_path = ::pcu_alias)]
        fn $flat<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
            let id = pcu::context::global_invocation_id();
            let original = input[id];
            let factor = *seed;
            let shifted = original + factor;
            let scaled = shifted * original;
            output[id] = scaled + original;
        }
        #[pcu(invocations = 3, crate_path = ::pcu_alias)]
        fn $grid<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
            let mut id = pcu::context::global_invocation_id();
            let stride = pcu::context::invocation_count();
            while id < N {
                output[id] = $helper(input[id], *seed);
                id += stride;
            }
        }
        #[pcu(invocations = 3, crate_path = ::pcu_alias)]
        fn $flat_grid<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
            let mut id = pcu::context::global_invocation_id();
            let stride = pcu::context::invocation_count();
            while id < N {
                let original = input[id];
                let factor = *seed;
                let shifted = original + factor;
                let scaled = shifted * original;
                output[id] = scaled + original;
                id += stride;
            }
        }
    };
}
sources!(f32, shifted, direct32, flat32, grid32, flat_grid32);
sources!(f64, helper_alias, direct64, flat64, grid64, flat_grid64);

#[test]
fn typed_saved_helper_locals_have_identical_inline_ir_for_all_normal_requests() {
    macro_rules! compare {
        ($source_bindings:ident, $source_ir:ident, $flat_bindings:ident, $flat_ir:ident) => {
            for request in super::typed::requests() {
                let source_bindings = $source_bindings();
                let flat_bindings = $flat_bindings();
                $source_ir::<47>(
                    &source_bindings,
                    request.float_underflow,
                    request.range_policy,
                    request,
                )
                .unwrap()
                .with_ir(|source| {
                    $flat_ir::<47>(
                        &flat_bindings,
                        request.float_underflow,
                        request.range_policy,
                        request,
                    )
                    .unwrap()
                    .with_ir(|flat| {
                        assert_eq!(source.ops, flat.ops);
                        assert_eq!(source.bindings, flat.bindings);
                        assert_eq!(source.numerical_requirements, flat.numerical_requirements);
                        pcu_alias::validate_typed_dispatch_value_flow(source).unwrap();
                    });
                });
            }
        };
    }
    compare!(
        direct32_bindings,
        __direct32_ir_with_float_underflow_policy,
        flat32_bindings,
        __flat32_ir_with_float_underflow_policy
    );
    compare!(
        grid32_bindings,
        __grid32_ir_with_float_underflow_policy,
        flat_grid32_bindings,
        __flat_grid32_ir_with_float_underflow_policy
    );
    compare!(
        direct64_bindings,
        __direct64_ir_with_float_underflow_policy,
        flat64_bindings,
        __flat64_ir_with_float_underflow_policy
    );
    compare!(
        grid64_bindings,
        __grid64_ir_with_float_underflow_policy,
        flat_grid64_bindings,
        __flat_grid64_ir_with_float_underflow_policy
    );
}

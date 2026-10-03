//! Concrete helper composition must equal independently authored checked integer IR.
use fusion_pcu_macros::pcu;

macro_rules! family {
    ($module:ident, $ty:ident) => {
        mod $module {
            use super::*;

            #[pcu(crate_path = ::pcu_alias)]
            #[allow(clippy::missing_const_for_fn)] // PCU scalar companions have a non-const lowering contract.
            fn helper(mut value: $ty, seed: $ty) -> $ty {
                let original: $ty = value;
                value += seed;
                value *= original;
                value - original
            }

            #[pcu(crate_path = ::pcu_alias)]
            fn nested(value: $ty, seed: $ty) -> $ty {
                let result = helper(value, seed);
                result
            }

            #[pcu(invocations = N, crate_path = ::pcu_alias)]
            fn direct<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = nested(input[id], *seed);
            }

            #[pcu(invocations = N, crate_path = ::pcu_alias)]
            fn flat<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                let original = input[id];
                let factor = *seed;
                let shifted = original + factor;
                let scaled = shifted * original;
                output[id] = scaled - original;
            }

            #[pcu(invocations = 3, crate_path = ::pcu_alias)]
            fn grid<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let mut id = pcu::context::global_invocation_id();
                let stride = pcu::context::invocation_count();
                while id < N {
                    output[id] = nested(input[id], *seed);
                    id += stride;
                }
            }

            #[pcu(invocations = 3, crate_path = ::pcu_alias)]
            fn flat_grid<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let mut id = pcu::context::global_invocation_id();
                let stride = pcu::context::invocation_count();
                while id < N {
                    let original = input[id];
                    let factor = *seed;
                    let shifted = original + factor;
                    let scaled = shifted * original;
                    output[id] = scaled - original;
                    id += stride;
                }
            }

            pub fn verify() {
                macro_rules! compare {
                    ($source_bindings:ident, $source:ident, $flat_bindings:ident, $flat:ident) => {
                        for request in super::super::super::typed::requests() {
                            let source_bindings = $source_bindings();
                            let flat_bindings = $flat_bindings();
                            $source::<47>(
                                &source_bindings,
                                request.float_underflow,
                                request.range_policy,
                                request,
                            )
                            .unwrap()
                            .with_ir(|source| {
                                $flat::<47>(
                                    &flat_bindings,
                                    request.float_underflow,
                                    request.range_policy,
                                    request,
                                )
                                .unwrap()
                                .with_ir(|flat| {
                                    assert_eq!(source.ops, flat.ops);
                                    assert_eq!(source.bindings, flat.bindings);
                                    assert_eq!(
                                        source.numerical_requirements,
                                        flat.numerical_requirements
                                    );
                                    pcu_alias::validate_typed_dispatch_value_flow(source).unwrap();
                                });
                            });
                        }
                    };
                }
                compare!(
                    direct_bindings,
                    __direct_ir_with_float_underflow_policy,
                    flat_bindings,
                    __flat_ir_with_float_underflow_policy
                );
                compare!(
                    grid_bindings,
                    __grid_ir_with_float_underflow_policy,
                    flat_grid_bindings,
                    __flat_grid_ir_with_float_underflow_policy
                );
            }
        }
    };
}
family!(unsigned8, u8);
family!(unsigned16, u16);
family!(unsigned32, u32);
family!(unsigned64, u64);
family!(unsigned128, u128);
family!(signed8, i8);
family!(signed16, i16);
family!(signed32, i32);
family!(signed64, i64);
family!(signed128, i128);

#[test]
fn ten_integer_helper_profiles_match_inline_ir_for_all_normal_requests() {
    unsigned8::verify();
    unsigned16::verify();
    unsigned32::verify();
    unsigned64::verify();
    unsigned128::verify();
    signed8::verify();
    signed16::verify();
    signed32::verify();
    signed64::verify();
    signed128::verify();
}

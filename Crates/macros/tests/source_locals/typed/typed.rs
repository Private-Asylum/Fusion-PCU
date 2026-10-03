//! Explicit local types preserve the inferred program, including saved SSA values.
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    PcuScalar,
    PcuCheckedFloat,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn typed_direct<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original: T = input[id];
    let mut value: T = original;
    let saved: _ = value;
    value = *seed;
    stage[id] = value;
    output[id] = saved;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn inferred_direct<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    let mut value = original;
    let saved = value;
    value = *seed;
    stage[id] = value;
    output[id] = saved;
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn typed_grid<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let original: T = input[id];
        let mut value: T = original;
        let saved: _ = value;
        value = *seed;
        stage[id] = value;
        output[id] = saved;
        id += stride;
    }
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn inferred_grid<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let original = input[id];
        let mut value = original;
        let saved = value;
        value = *seed;
        stage[id] = value;
        output[id] = saved;
        id += stride;
    }
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn typed_float<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let mut value: T = input[id];
    let original: _ = value;
    value = value + value;
    output[id] = value * original;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn inferred_float<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let mut value = input[id];
    let original = value;
    value = value + value;
    output[id] = value * original;
}

pub fn requests() -> impl Iterator<Item = PcuImplementationRequirements> {
    [PcuNumericalMode::Boundary, PcuNumericalMode::Strict]
        .into_iter()
        .flat_map(|numerical_mode| {
            [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ]
            .into_iter()
            .flat_map(move |compound_arithmetic| {
                [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ]
                .into_iter()
                .flat_map(move |precision| {
                    [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    ]
                    .into_iter()
                    .flat_map(move |float_underflow| {
                        [PcuRangePolicy::Reject, PcuRangePolicy::Clamp]
                            .into_iter()
                            .map(move |range_policy| PcuImplementationRequirements {
                                numerical_mode,
                                numerical_options: PcuNumericalOptions {
                                    compound_arithmetic,
                                    precision,
                                    reproducibility: PcuReproducibility::Unspecified,
                                },
                                float_underflow,
                                range_policy,
                            })
                    })
                })
            })
        })
}
fn exact_transport<T: PcuScalar>() {
    let typed = typed_direct_bindings::<T>();
    let inferred = inferred_direct_bindings::<T>();
    let typed_grid = typed_grid_bindings::<T>();
    let inferred_grid = inferred_grid_bindings::<T>();
    for req in requests() {
        __typed_direct_ir_with_float_underflow_policy::<T, 47>(
            &typed,
            req.float_underflow,
            req.range_policy,
            req,
        )
        .unwrap()
        .with_ir(|lhs| {
            __inferred_direct_ir_with_float_underflow_policy::<T, 47>(
                &inferred,
                req.float_underflow,
                req.range_policy,
                req,
            )
            .unwrap()
            .with_ir(|rhs| {
                assert_eq!(lhs.ops, rhs.ops);
                assert_eq!(lhs.bindings, rhs.bindings);
                assert_eq!(lhs.numerical_requirements, rhs.numerical_requirements);
            });
        });
        __typed_grid_ir_with_float_underflow_policy::<T, 47>(
            &typed_grid,
            req.float_underflow,
            req.range_policy,
            req,
        )
        .unwrap()
        .with_ir(|lhs| {
            __inferred_grid_ir_with_float_underflow_policy::<T, 47>(
                &inferred_grid,
                req.float_underflow,
                req.range_policy,
                req,
            )
            .unwrap()
            .with_ir(|rhs| {
                assert_eq!(lhs.ops, rhs.ops);
                assert_eq!(lhs.bindings, rhs.bindings);
                assert_eq!(lhs.numerical_requirements, rhs.numerical_requirements);
            });
        });
    }
}
#[test]
fn twenty_two_carrier_annotations_have_identical_inferred_ir() {
    macro_rules! types { ($($ty:ty),+) => { $(exact_transport::<$ty>();)+ }; }
    types!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        PcuU256,
        PcuI256,
        PcuU512,
        PcuI512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
}
fn exact_float<T: PcuCheckedFloat>() {
    let typed = typed_float_bindings::<T>();
    let inferred = inferred_float_bindings::<T>();
    for req in requests() {
        __typed_float_ir_with_float_underflow_policy::<T, 47>(
            &typed,
            req.float_underflow,
            req.range_policy,
            req,
        )
        .unwrap()
        .with_ir(|lhs| {
            __inferred_float_ir_with_float_underflow_policy::<T, 47>(
                &inferred,
                req.float_underflow,
                req.range_policy,
                req,
            )
            .unwrap()
            .with_ir(|rhs| {
                assert_eq!(lhs.ops, rhs.ops);
                assert_eq!(lhs.bindings, rhs.bindings);
                assert_eq!(lhs.numerical_requirements, rhs.numerical_requirements);
            });
        });
    }
}
#[test]
fn six_float_annotations_preserve_every_checked_instruction_policy() {
    macro_rules! types { ($($ty:ty),+) => { $(exact_float::<$ty>();)+ }; }
    types!(
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64
    );
}

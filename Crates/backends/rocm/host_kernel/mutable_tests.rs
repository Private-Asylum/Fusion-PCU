//! Mutable spelling retains the exact qualified ordered-map operation stream.
#[rustfmt::skip]
use fusion_pcu::{
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[path = "../benches/mutable_ordered_float_maps/source/source.rs"]
#[allow(dead_code)] // Both source entries are separately exercised by their benchmark.
mod mutation;
use super::staging_tests::source as original;
fn compare<T: pcu_facade::PcuCheckedFloat>(request: PcuImplementationRequirements) {
    let bindings = original::direct_bindings::<T>();
    assert_eq!(bindings, mutation::direct_bindings::<T>());
    let original = original::__direct_ir_with_float_underflow_policy::<T, 65>(
        &bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap();
    let mutation = mutation::__direct_ir_with_float_underflow_policy::<T, 65>(
        &bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap();
    let before = original.ir();
    let after = mutation.ir();
    assert_eq!(before.ops, after.ops);
    assert_eq!(before.bindings, after.bindings);
    assert_eq!(before.entry.logical_shape, after.entry.logical_shape);
    assert_eq!(before.numerical_requirements, after.numerical_requirements);
    let original = original::__grid_ir_with_float_underflow_policy::<T, 65>(
        &bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap();
    let mutation = mutation::__grid_ir_with_float_underflow_policy::<T, 65>(
        &bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap();
    original.with_ir(|before| {
        mutation.with_ir(|after| {
            assert_eq!(before.ops, after.ops);
            assert_eq!(before.bindings, after.bindings);
            assert_eq!(before.entry.logical_shape, after.entry.logical_shape);
            assert_eq!(before.numerical_requirements, after.numerical_requirements);
        });
    });
}
#[test]
fn mutable_ordered_source_preserves_exact_operations_and_policy() {
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        let mut request = PcuImplementationRequirements {
                            numerical_mode,
                            float_underflow,
                            range_policy,
                            ..PcuImplementationRequirements::DEFAULT
                        };
                        request.numerical_options.compound_arithmetic = compound;
                        request.numerical_options.precision = precision;
                        compare::<PcuF16Bits>(request);
                        compare::<PcuBf16Bits>(request);
                        compare::<PcuF8E4M3FnBits>(request);
                        compare::<PcuF8E5M2Bits>(request);
                        compare::<f32>(request);
                        compare::<f64>(request);
                    }
                }
            }
        }
    }
}

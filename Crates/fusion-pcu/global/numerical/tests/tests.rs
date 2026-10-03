//! Eligibility neither opens a runtime nor replaces normal schema validation.
use super::*;
#[rustfmt::skip]
use crate::{
    assess_checked_float_map_resources,
    describe_portable_v1_unary_map,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuCompoundArithmeticPolicy,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuKernelId,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuValueType,
    PcuValueTypeCaps,
};

#[crate::pcu(invocations = N, crate_path = crate, flag(deterministic))]
fn portable_composed_resources<T: PcuCheckedFloat, const N: usize>(
    _unused_input: &[T],
    output: &mut [T],
    factor: &T,
    input: &[T],
    _unused_output: &mut [T],
) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[0]) * *factor;
}

#[crate::pcu(invocations = 3, crate_path = crate, flag(deterministic))]
fn portable_composed_grid_resources<T: PcuCheckedFloat, const N: usize>(
    _unused_input: &[T],
    output: &mut [T],
    factor: &T,
    input: &[T],
    _unused_output: &mut [T],
) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = (input[id] + input[0]) * *factor;
        id += stride;
    }
}

fn composed_resource_descriptor_does_not_admit_execution<T: PcuCheckedFloat>() {
    let check = |kernel: &PcuDispatchKernelIr<'_>| {
        let schema = assess_checked_float_map_resources::<3>(
            kernel,
            PcuValueType::Scalar(T::TYPE),
            PcuValueTypeCaps::for_scalar(T::TYPE),
        )
        .unwrap();
        assert_eq!(schema.requirements, kernel.numerical_requirements);
        assert_eq!(schema.logical_extent, 19);
        assert_eq!(schema.resources().len(), 3);
        let input = schema.resource(PcuBindingRef::new(0, 3)).unwrap();
        assert_eq!(input.minimum_read_elements, 19);
        assert_eq!(input.minimum_write_elements, 0);
        let factor = schema.resource(PcuBindingRef::new(0, 2)).unwrap();
        assert_eq!(factor.minimum_read_elements, 1);
        assert_eq!(factor.minimum_write_elements, 0);
        let output = schema.resource(PcuBindingRef::new(0, 1)).unwrap();
        assert_eq!(output.minimum_read_elements, 0);
        assert_eq!(output.minimum_write_elements, 19);
        assert!(schema.resource(PcuBindingRef::new(0, 0)).is_none());
        assert!(schema.resource(PcuBindingRef::new(0, 4)).is_none());
        assert_eq!(kernel.bindings[4].access, PcuBindingAccess::ReadWrite);
        let options = kernel.numerical_requirements.numerical_options;
        assert!(matches!(
            validate_invocation_contract(kernel),
            Err(PcuExecutionError::UnsupportedNumericalOptions(actual)) if actual == options,
        ));
    };
    let bindings = portable_composed_resources_bindings::<T>();
    let lowered = portable_composed_resources_ir::<T, 19>(&bindings).unwrap();
    assert_eq!(lowered.ir().entry.logical_shape, [19, 1, 1]);
    check(&lowered.ir());
    let bindings = portable_composed_grid_resources_bindings::<T>();
    portable_composed_grid_resources_ir::<T, 19>(&bindings)
        .unwrap()
        .with_ir(|kernel| {
            assert_eq!(kernel.entry.logical_shape, [3, 1, 1]);
            check(kernel);
        });
}

#[test]
fn genuine_composed_source_resources_preserve_ignored_mutable_arguments_and_admission_gate() {
    composed_resource_descriptor_does_not_admit_execution::<PcuF16Bits>();
    composed_resource_descriptor_does_not_admit_execution::<PcuBf16Bits>();
    composed_resource_descriptor_does_not_admit_execution::<PcuF8E4M3FnBits>();
    composed_resource_descriptor_does_not_admit_execution::<PcuF8E5M2Bits>();
    composed_resource_descriptor_does_not_admit_execution::<f32>();
    composed_resource_descriptor_does_not_admit_execution::<f64>();
}

#[crate::pcu(invocations = N, crate_path = crate, flag(deterministic))]
fn portable_flip<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

#[crate::pcu(invocations = 3, crate_path = crate, flag(deterministic), flag(clamp_range), flag(reject_subnormal_result))]
fn portable_activate<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = pcu::relu(input[0]);
        id += stride;
    }
}

fn unary_descriptor_preserves_cold_eligibility<T: PcuCheckedFloat>() {
    let bindings = portable_flip_bindings::<T>();
    let ir = portable_flip_ir::<T, 19>(&bindings).unwrap();
    let check = |kernel: &PcuDispatchKernelIr<'_>| {
        let description = describe_portable_v1_unary_map(kernel).unwrap();
        assert_eq!(description.scalar, T::TYPE);
        assert_eq!(description.requirements, kernel.numerical_requirements);
        // This gate is necessary structure only. No runtime or provider is
        // opened here; an exact qualified offer remains mandatory downstream.
        validate_invocation_contract(kernel).unwrap();
    };
    check(&ir.ir());
    let bindings = portable_activate_bindings::<T>();
    portable_activate_ir::<T, 19>(&bindings)
        .unwrap()
        .with_ir(|kernel| {
            let description = describe_portable_v1_unary_map(kernel).unwrap();
            assert!(description.broadcast_input);
            assert_eq!(description.input_extent(), 1);
            assert_eq!(description.submitted_invocations, 3);
            assert_eq!(description.logical_extent, 19);
            check(kernel);
        });
}

#[test]
fn genuine_portable_unary_source_remains_separate_from_backend_opt_in() {
    unary_descriptor_preserves_cold_eligibility::<PcuF16Bits>();
    unary_descriptor_preserves_cold_eligibility::<PcuBf16Bits>();
    unary_descriptor_preserves_cold_eligibility::<PcuF8E4M3FnBits>();
    unary_descriptor_preserves_cold_eligibility::<PcuF8E5M2Bits>();
    unary_descriptor_preserves_cold_eligibility::<f32>();
    unary_descriptor_preserves_cold_eligibility::<f64>();
}

#[test]
fn unary_eligibility_requires_original_header_and_checked_operation_agreement() {
    let bindings = portable_flip_bindings::<f32>();
    let lowered = portable_flip_ir::<f32, 19>(&bindings).unwrap();
    let mut kernel = lowered.ir();
    kernel.numerical_requirements.float_underflow =
        crate::PcuFloatUnderflowPolicy::AllowGradualUnderflow;
    let options = kernel.numerical_requirements.numerical_options;
    assert!(matches!(validate_invocation_contract(&kernel),
        Err(PcuExecutionError::UnsupportedNumericalOptions(actual)) if actual == options));
}

fn unproved() -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(37),
        entry: PcuDispatchEntryPoint {
            name: "unproved-numerical-profile",
            logical_shape: [1, 1, 1],
        },
        bindings: &[],
        ports: &[],
        parameters: &[],
        ops: &[],
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}

#[test]
fn unspecified_reproducibility_preserves_backend_schema_validation() {
    let mut kernel = unproved();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                kernel.numerical_requirements.numerical_mode = mode;
                kernel
                    .numerical_requirements
                    .numerical_options
                    .compound_arithmetic = compound;
                kernel.numerical_requirements.numerical_options.precision = precision;
                // Success is only numerical eligibility. This empty body still
                // needs ordinary typed/resource/schema validation by a backend.
                validate_invocation_contract(&kernel).unwrap();
            }
        }
    }
}

#[test]
fn unproved_portable_request_retains_original_error_options() {
    let mut kernel = unproved();
    kernel
        .numerical_requirements
        .numerical_options
        .reproducibility = PcuReproducibility::PortableV1;
    kernel
        .numerical_requirements
        .numerical_options
        .compound_arithmetic = PcuCompoundArithmeticPolicy::BackendDefined;
    kernel.numerical_requirements.numerical_options.precision =
        PcuPrecisionPolicy::BackendOptimized;
    let expected = kernel.numerical_requirements.numerical_options;
    assert!(matches!(
        validate_invocation_contract(&kernel),
        Err(PcuExecutionError::UnsupportedNumericalOptions(actual)) if actual == expected,
    ));
}

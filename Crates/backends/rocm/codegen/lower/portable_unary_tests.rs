//! Exact bounded Portable unary admission; emitted checked bodies remain unchanged.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuCompoundArithmeticPolicy,
    PcuDispatchEntryPoint,
    PcuDispatchFloatUnaryOp,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuReproducibility,
};
const INPUT: PcuBindingRef = PcuBindingRef::new(7, 9);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(2, 3);
fn declarations(scalar: PcuScalarType) -> [PcuBinding<'static>; 3] {
    [
        (PcuBindingRef::new(5, 4), PcuBindingAccess::ReadOnly),
        (OUTPUT, PcuBindingAccess::ReadWrite),
        (INPUT, PcuBindingAccess::ReadOnly),
    ]
    .map(|(binding, access)| {
        PcuBinding::value(
            None,
            binding.set,
            binding.binding,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    })
}
fn instructions(
    scalar: PcuScalarType,
    operation: PcuDispatchFloatUnaryOp,
    index: PcuDispatchIndex,
    requirements: PcuImplementationRequirements,
) -> [PcuDispatchOp<'static>; 4] {
    [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(0),
            binding: INPUT,
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            result: PcuDispatchValueId(1),
            value: PcuDispatchValueId(0),
            value_type: PcuValueType::Scalar(scalar),
            op: operation,
            range_policy: requirements.range_policy,
            underflow_policy: requirements.float_underflow,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index: if index == PcuDispatchIndex::BindingElementZero {
                PcuDispatchIndex::InvocationId
            } else {
                index
            },
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ]
}
fn kernel<'a>(
    scalar: PcuScalarType,
    requirements: PcuImplementationRequirements,
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [PcuDispatchOp<'a>],
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: requirements,
        id: PcuKernelId(0x5055_4e59),
        entry: PcuDispatchEntryPoint {
            name: "portable_unary",
            logical_shape: [23, 1, 1],
        },
        bindings,
        ops,
        ports: &[],
        parameters: &[],
        type_caps: PcuValueTypeCaps::for_scalar(scalar),
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES)
            .union(PcuDispatchFeatureCaps::RANGE_CLAMP),
    }
}
fn accepted(
    scalar: PcuScalarType,
    op: PcuDispatchFloatUnaryOp,
    requirements: PcuImplementationRequirements,
) {
    let bindings = declarations(scalar);
    for index in [
        PcuDispatchIndex::InvocationId,
        PcuDispatchIndex::GridStrideId,
        PcuDispatchIndex::BindingElementZero,
    ] {
        let body = instructions(scalar, op, index, requirements);
        let outer = [
            PcuDispatchOp::GridStrideLoop {
                extent: 65,
                body: &body[..3],
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let ir = kernel(
            scalar,
            requirements,
            &bindings,
            if index == PcuDispatchIndex::GridStrideId {
                &outer
            } else {
                &body
            },
        );
        let descriptor = fusion_pcu::describe_portable_v1_unary_map(&ir).unwrap();
        assert_eq!(descriptor.requirements, requirements);
        assert_eq!(descriptor.input_binding, INPUT);
        assert_eq!(descriptor.output_binding, OUTPUT);
        assert_eq!(
            descriptor.broadcast_input,
            index == PcuDispatchIndex::BindingElementZero
        );
        let projection = map_binding_projection(&ir)
            .expect("unary actual resources must exclude the unread declaration");
        assert_eq!(projection.input_bindings(), &[INPUT]);
        assert!(projection.contains_output(OUTPUT));
        assert!(!projection.contains_output(bindings[0].reference()));
        let portable = lower_dispatch_to_hip_source(&ir).unwrap();
        assert!(!portable.contains("binding_5_4"));
        let mut normal = ir;
        normal
            .numerical_requirements
            .numerical_options
            .reproducibility = PcuReproducibility::Unspecified;
        let normal = lower_dispatch_to_hip_source(&normal).unwrap();
        assert_eq!(
            portable.split_once('\n').unwrap().1,
            normal.split_once('\n').unwrap().1
        );
        assert!(crate::owned_dispatch::checked_scalar_fault_law(&ir).is_some());
    }
}
#[test]
fn portable_unary_six_formats_all_tuples_keep_exact_request_and_checked_body() {
    let mut count = 0;
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        for op in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
            for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for compound_arithmetic in [
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
                                let mut requirements = PcuImplementationRequirements {
                                    numerical_mode,
                                    float_underflow,
                                    range_policy,
                                    ..PcuImplementationRequirements::DEFAULT
                                };
                                requirements.numerical_options.compound_arithmetic =
                                    compound_arithmetic;
                                requirements.numerical_options.precision = precision;
                                requirements.numerical_options.reproducibility =
                                    PcuReproducibility::PortableV1;
                                accepted(scalar, op, requirements);
                                count += 3;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(count, 1728);
}
#[test]
fn portable_unary_rejects_header_mismatch_mutable_input_and_bad_ssa() {
    let scalar = PcuScalarType::F16;
    let mut requirements = PcuImplementationRequirements::DEFAULT;
    requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    let bindings = declarations(scalar);
    let mut body = instructions(
        scalar,
        PcuDispatchFloatUnaryOp::Neg,
        PcuDispatchIndex::InvocationId,
        requirements,
    );
    let mut ir = kernel(scalar, requirements, &bindings, &body);
    ir.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
    assert!(lower_dispatch_to_hip_source(&ir).is_err());
    ir.numerical_requirements = requirements;
    ir.numerical_requirements.float_underflow = PcuFloatUnderflowPolicy::RejectSubnormalResult;
    assert!(lower_dispatch_to_hip_source(&ir).is_err());
    let mut writable = bindings;
    writable[2].access = PcuBindingAccess::ReadWrite;
    assert!(lower_dispatch_to_hip_source(&kernel(scalar, requirements, &writable, &body)).is_err());
    if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { value, .. }) = &mut body[2] {
        *value = PcuDispatchValueId(0);
    }
    assert!(lower_dispatch_to_hip_source(&kernel(scalar, requirements, &bindings, &body)).is_err());
}

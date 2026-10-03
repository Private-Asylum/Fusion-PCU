//! Eligibility is exact structural law, independently of any executor's portable proof.
use super::*;
#[rustfmt::skip]
use crate::{
    model::PcuDispatchKernelBuilder,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchValueId,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuPrecisionPolicy,
};
const FORMATS: [PcuScalarType; 4] = [
    PcuScalarType::F16,
    PcuScalarType::BF16,
    PcuScalarType::F8E4M3FN,
    PcuScalarType::F8E5M2,
];
const OPS: [PcuDispatchFloatBinaryOp; 4] = [
    PcuDispatchFloatBinaryOp::Add,
    PcuDispatchFloatBinaryOp::Sub,
    PcuDispatchFloatBinaryOp::Mul,
    PcuDispatchFloatBinaryOp::Div,
];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
];
fn bindings(scalar: PcuScalarType) -> [PcuBinding<'static>; 3] {
    core::array::from_fn(|index| {
        PcuBinding::value(
            None,
            0,
            u32::try_from(index).unwrap(),
            PcuBindingStorageClass::Storage,
            if index == 2 {
                PcuBindingAccess::ReadWrite
            } else {
                PcuBindingAccess::ReadOnly
            },
            PcuValueType::Scalar(scalar),
        )
    })
}
fn body(
    scalar: PcuScalarType,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
    grid: bool,
    broadcast: bool,
    operands: [u16; 2],
) -> [PcuDispatchOp<'static>; 4] {
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index: if broadcast {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: PcuValueType::Scalar(scalar),
            op,
            underflow_policy: policy,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(operands[0]),
            rhs: PcuDispatchValueId(operands[1]),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: PcuDispatchValueId(3),
        }),
    ]
}
fn requirements(policy: PcuFloatUnderflowPolicy) -> PcuImplementationRequirements {
    let mut requirements = PcuImplementationRequirements::DEFAULT;
    requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    requirements.float_underflow = policy;
    requirements
}
fn builder<'a>(
    bindings: &'a [PcuBinding<'a>],
    body: &[PcuDispatchOp<'a>],
    requirements: PcuImplementationRequirements,
) -> PcuDispatchKernelBuilder<'a, 5> {
    PcuDispatchKernelBuilder::new(1, "portable", [3, 1, 1])
        .with_bindings(bindings)
        .with_numerical_requirements(requirements)
        .with_ops(body)
        .unwrap()
        .with_control_op(PcuDispatchControlOp::Return)
        .unwrap()
}
#[test]
fn all_formats_operations_policies_and_independent_permissions_describe_exact_maps() {
    let mut checked = 0;
    for scalar in FORMATS {
        for op in OPS {
            for policy in POLICIES {
                let bindings = bindings(scalar);
                for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                    for compound in [
                        PcuCompoundArithmeticPolicy::Checked,
                        PcuCompoundArithmeticPolicy::BackendDefined,
                    ] {
                        for precision in [
                            PcuPrecisionPolicy::Preserve,
                            PcuPrecisionPolicy::BackendOptimized,
                        ] {
                            let mut requirements = requirements(policy);
                            requirements.numerical_mode = mode;
                            requirements.numerical_options.compound_arithmetic = compound;
                            requirements.numerical_options.precision = precision;
                            for broadcast in [false, true] {
                                for operands in [[1, 2], [2, 1], [1, 1], [2, 2]] {
                                    let body = body(scalar, op, policy, false, broadcast, operands);
                                    let builder = builder(&bindings, &body, requirements);
                                    let actual = describe_portable_v1_map(&builder.ir()).unwrap();
                                    assert_eq!(actual.scalar, scalar);
                                    assert_eq!(actual.operation, op);
                                    assert_eq!(actual.underflow, policy);
                                    assert_eq!(actual.logical_extent, 3);
                                    assert_eq!(actual.broadcast_loads, [false, broadcast]);
                                    assert_eq!(
                                        actual.operands,
                                        operands.map(|id| u8::try_from(id - 1).unwrap())
                                    );
                                    assert_eq!(actual.output_binding, PcuBindingRef::new(0, 2));
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 3072);
}
#[test]
fn canonical_grid_stride_retains_extent_broadcast_and_actual_operands() {
    for scalar in FORMATS {
        for op in OPS {
            for policy in POLICIES {
                let bindings = bindings(scalar);
                let body = body(scalar, op, policy, true, true, [2, 1]);
                let ops = [
                    PcuDispatchOp::GridStrideLoop {
                        body: &body,
                        extent: 17,
                    },
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                ];
                let builder = PcuDispatchKernelBuilder::<2>::new(1, "grid", [3, 1, 1])
                    .with_bindings(&bindings)
                    .with_numerical_requirements(requirements(policy))
                    .with_ops(&ops)
                    .unwrap();
                let actual = describe_portable_v1_map(&builder.ir()).unwrap();
                assert_eq!(actual.logical_extent, 17);
                assert_eq!(actual.submitted_invocations, 3);
                assert_eq!(actual.operands, [1, 0]);
                assert_eq!(actual.broadcast_loads, [false, true]);
            }
        }
    }
}
#[test]
fn unsupported_numerics_and_malformed_schema_do_not_become_portable() {
    let policy = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let bindings = bindings(PcuScalarType::F16);
    let mut body = body(
        PcuScalarType::F16,
        PcuDispatchFloatBinaryOp::Div,
        policy,
        false,
        false,
        [1, 2],
    );
    let builder = builder(&bindings, &body, requirements(policy));
    let kernel = builder.ir();
    let mut changed = kernel;
    changed
        .numerical_requirements
        .numerical_options
        .reproducibility = PcuReproducibility::Unspecified;
    assert_eq!(
        describe_portable_v1_map(&changed),
        Err(PcuPortableV1MapError::NotRequested)
    );
    changed = kernel;
    changed.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
    assert_eq!(
        describe_portable_v1_map(&changed),
        Err(PcuPortableV1MapError::UnsupportedRange)
    );
    changed = kernel;
    changed.numerical_requirements.float_underflow = PcuFloatUnderflowPolicy::AllowGradualUnderflow;
    assert_eq!(
        describe_portable_v1_map(&changed),
        Err(PcuPortableV1MapError::UnderflowMismatch)
    );
    for shape in [[0, 1, 1], [3, 2, 1], [3, 1, 2]] {
        changed = kernel;
        changed.entry.logical_shape = shape;
        assert_eq!(
            describe_portable_v1_map(&changed),
            Err(PcuPortableV1MapError::InvalidLogicalShape(shape))
        );
    }
    changed = kernel;
    changed.bindings = &bindings[..2];
    assert!(describe_portable_v1_map(&changed).is_err());
    if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { value, .. }) = &mut body[3] {
        *value = PcuDispatchValueId(4);
    }
    let malformed = self::builder(&bindings, &body, requirements(policy));
    assert!(matches!(
        describe_portable_v1_map(&malformed.ir()),
        Err(PcuPortableV1MapError::InvalidValueFlow(_))
    ));
    for scalar in [
        PcuScalarType::F32,
        PcuScalarType::F64,
        PcuScalarType::F128,
        PcuScalarType::F256,
    ] {
        let bindings = self::bindings(scalar);
        let body = self::body(
            scalar,
            PcuDispatchFloatBinaryOp::Add,
            policy,
            false,
            false,
            [1, 2],
        );
        let builder = self::builder(&bindings, &body, requirements(policy));
        assert_eq!(
            describe_portable_v1_map(&builder.ir()),
            Err(PcuPortableV1MapError::UnsupportedScalar)
        );
    }
}

//! Eligibility proofs are deliberately separate from hardware conformance.
use super::*;
#[rustfmt::skip]
use crate::{
    IntegerMapValidationError,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex as Index,
    PcuDispatchValueId as Id,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
};
use crate::model::PcuIntegerDivFlags;

fn bindings(scalar: PcuScalarType) -> [PcuBinding<'static>; 4] {
    core::array::from_fn(|slot| {
        PcuBinding::value(
            None,
            0,
            u32::try_from(slot).unwrap(),
            PcuBindingStorageClass::Storage,
            if slot < 2 {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::ReadWrite
            },
            PcuValueType::Scalar(scalar),
        )
    })
}

fn body(scalar: PcuScalarType, index: Index) -> [PcuDispatchOp<'static>; 5] {
    use PcuDispatchDataOp as Data;
    use PcuDispatchOp::Data as Op;
    [
        Op(Data::BindingLoad {
            result: Id(600),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        Op(Data::BindingLoad {
            result: Id(601),
            binding: PcuBindingRef::new(0, 1),
            index,
        }),
        Op(Data::CheckedDivRem {
            value_type: PcuValueType::Scalar(scalar),
            flags: PcuIntegerDivFlags::CHECKED,
            quotient: Id(65534),
            remainder: Id(65535),
            lhs: Id(600),
            rhs: Id(601),
        }),
        Op(Data::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: Id(65534),
        }),
        Op(Data::BindingStore {
            binding: PcuBindingRef::new(0, 3),
            index,
            value: Id(65535),
        }),
    ]
}

fn kernel<'a>(
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [PcuDispatchOp<'a>],
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: crate::PcuImplementationRequirements {
            numerical_options: PcuNumericalOptions {
                reproducibility: PcuReproducibility::PortableV1,
                ..Default::default()
            },
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
        id: PcuKernelId(91),
        entry: PcuDispatchEntryPoint {
            name: "portable-div-rem",
            logical_shape: [3, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}

fn assert_roles(
    scalar: PcuScalarType,
    mode: PcuNumericalMode,
    compound: PcuCompoundArithmeticPolicy,
    precision: PcuPrecisionPolicy,
    underflow: PcuFloatUnderflowPolicy,
) {
    let original = bindings(scalar);
    let declared = [original[3], original[0], original[2], original[1]];
    let single = [original[3], original[0], original[2]];
    for grid in [false, true] {
        let index = if grid {
            Index::GridStrideId
        } else {
            Index::InvocationId
        };
        for repeated in [false, true] {
            for reverse in [false, true] {
                let mut body = body(scalar, index);
                let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. }) =
                    &mut body[1]
                else {
                    unreachable!()
                };
                if repeated {
                    *binding = original[0].reference();
                }
                *index = Index::BindingElementZero;
                let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { lhs, rhs, .. }) =
                    &mut body[2]
                else {
                    unreachable!()
                };
                if reverse {
                    core::mem::swap(lhs, rhs);
                }
                body.swap(3, 4);
                let direct = [
                    body[0],
                    body[1],
                    body[2],
                    body[3],
                    body[4],
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                ];
                let loops = [
                    PcuDispatchOp::GridStrideLoop {
                        extent: 19,
                        body: &body,
                    },
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
                ];
                for bindings in [&declared[..], &single[..]] {
                    if !repeated && bindings.len() == 3 {
                        continue;
                    }
                    let mut ir = kernel(bindings, if grid { &loops } else { &direct });
                    ir.numerical_requirements.numerical_mode = mode;
                    ir.numerical_requirements.float_underflow = underflow;
                    ir.numerical_requirements
                        .numerical_options
                        .compound_arithmetic = compound;
                    ir.numerical_requirements.numerical_options.precision = precision;
                    let described = describe_portable_v1_integer_div_rem_map(&ir).unwrap();
                    let extent = if grid { 19 } else { 3 };
                    assert_eq!(described.scalar, scalar);
                    assert_eq!(described.logical_extent, extent);
                    assert_eq!(described.submitted_invocations, 3);
                    assert_eq!(
                        described.operands.output_bindings(),
                        [original[2].reference(), original[3].reference()]
                    );
                    assert_eq!(
                        described
                            .operands
                            .input_element_counts(usize::try_from(extent).unwrap()),
                        if repeated {
                            [usize::try_from(extent).unwrap(), 0]
                        } else {
                            [usize::try_from(extent).unwrap(), 1]
                        }
                    );
                    assert_eq!(
                        described.operands.operand_inputs(),
                        if repeated {
                            [0, 0]
                        } else if reverse {
                            [1, 0]
                        } else {
                            [0, 1]
                        }
                    );
                }
            }
        }
    }
}

#[test]
fn all_fourteen_widths_retain_roles_across_independent_numerical_axes() {
    for scalar in [
        PcuScalarType::I8,
        PcuScalarType::U8,
        PcuScalarType::I16,
        PcuScalarType::U16,
        PcuScalarType::I32,
        PcuScalarType::U32,
        PcuScalarType::I64,
        PcuScalarType::U64,
        PcuScalarType::I128,
        PcuScalarType::U128,
        PcuScalarType::I256,
        PcuScalarType::U256,
        PcuScalarType::I512,
        PcuScalarType::U512,
    ] {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            for compound in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    for underflow in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    ] {
                        assert_roles(scalar, mode, compound, precision, underflow);
                    }
                }
            }
        }
    }
}

#[test]
fn unproved_policies_geometry_types_and_extra_effects_refuse() {
    let scalar = PcuScalarType::U512;
    let bindings = bindings(scalar);
    let body = body(scalar, Index::InvocationId);
    let direct = [
        body[0],
        body[1],
        body[2],
        body[3],
        body[4],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let original = kernel(&bindings, &direct);
    let mut ir = original;
    ir.numerical_requirements.numerical_options.reproducibility = PcuReproducibility::Unspecified;
    assert_eq!(
        describe_portable_v1_integer_div_rem_map(&ir),
        Err(Error::NotRequested)
    );
    ir = original;
    ir.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
    assert_eq!(
        describe_portable_v1_integer_div_rem_map(&ir),
        Err(Error::InvalidOperands(
            IntegerMapValidationError::UnsupportedRequirements
        ))
    );
    for shape in [[0, 1, 1], [3, 2, 1], [3, 1, 2]] {
        ir = original;
        ir.entry.logical_shape = shape;
        assert_eq!(
            describe_portable_v1_integer_div_rem_map(&ir),
            Err(Error::InvalidLogicalShape(shape))
        );
    }
    let mut total = direct;
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { flags, .. }) = &mut total[2] else {
        unreachable!()
    };
    *flags = PcuIntegerDivFlags::DIV_OR_ZERO;
    assert_eq!(
        describe_portable_v1_integer_div_rem_map(&kernel(&bindings, &total)),
        Err(Error::InvalidOperands(
            IntegerMapValidationError::UnsupportedOperation(2)
        ))
    );
    let mut float = direct;
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { value_type, .. }) = &mut float[2]
    else {
        unreachable!()
    };
    *value_type = PcuValueType::Scalar(PcuScalarType::F32);
    assert_eq!(
        describe_portable_v1_integer_div_rem_map(&kernel(&bindings, &float)),
        Err(Error::UnsupportedScalar)
    );
    let extra = [
        body[0], body[1], body[2], body[3], body[4], body[4], direct[5],
    ];
    assert_eq!(
        describe_portable_v1_integer_div_rem_map(&kernel(&bindings, &extra)),
        Err(Error::UnsupportedStructure)
    );
    let duplicate = [bindings[0], bindings[1], bindings[2], bindings[2]];
    assert_eq!(
        describe_portable_v1_integer_div_rem_map(&kernel(&duplicate, &direct)),
        Err(Error::InvalidOperands(
            IntegerMapValidationError::DuplicateBinding(bindings[2].reference())
        ))
    );
}

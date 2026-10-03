use super::*;
#[rustfmt::skip]
use crate::{
    describe_checked_float_unary_map,
    PcuBinding,
    PcuBindingAccess,
    PcuCheckedFloatUnaryMapDescription,
    PcuCheckedFloatUnaryMapError,
    PcuDispatchControlOp,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp as Unary,
    PcuDispatchIndex as Index,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuFloatUnderflowPolicy as Underflow,
    PcuKernelId,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy as Range,
    PcuScalarType as Scalar,
    PcuValueType,
    PcuValueTypeCaps,
};

type Error = PcuPortableV1UnaryMapError;
const INPUT: PcuBindingRef = PcuBindingRef::new(7, 9);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(2, 3);
const FORMATS: [Scalar; 6] = [
    Scalar::F16,
    Scalar::BF16,
    Scalar::F32,
    Scalar::F64,
    Scalar::F8E4M3FN,
    Scalar::F8E5M2,
];
const POLICIES: [Underflow; 3] = [
    Underflow::IeeeAfterRounding,
    Underflow::RejectSubnormalResult,
    Underflow::AllowGradualUnderflow,
];

fn bindings(scalar: Scalar) -> [PcuBinding<'static>; 3] {
    [
        (PcuBindingRef::new(5, 4), PcuBindingAccess::ReadOnly),
        (OUTPUT, PcuBindingAccess::WriteOnly),
        (INPUT, PcuBindingAccess::ReadOnly),
    ]
    .map(|(reference, access)| {
        PcuBinding::value(
            None,
            reference.set,
            reference.binding,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    })
}

fn body(
    scalar: Scalar,
    operation: Unary,
    index: Index,
    range: Range,
    underflow: Underflow,
) -> [Op<'static>; 4] {
    [
        Op::Data(Data::BindingLoad {
            result: Id(0),
            binding: INPUT,
            index,
        }),
        Op::Data(Data::CheckedFloatUnary {
            result: Id(1),
            value: Id(0),
            op: operation,
            value_type: PcuValueType::Scalar(scalar),
            range_policy: range,
            underflow_policy: underflow,
        }),
        Op::Data(Data::BindingStore {
            binding: OUTPUT,
            index: if index == Index::BindingElementZero {
                Index::InvocationId
            } else {
                index
            },
            value: Id(1),
        }),
        Op::Control(PcuDispatchControlOp::Return),
    ]
}

fn kernel<'a>(
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [Op<'a>],
    range: Range,
    underflow: Underflow,
) -> PcuDispatchKernelIr<'a> {
    let mut requirements = PcuDispatchKernelIr::DEFAULT_REQUIREMENTS;
    requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    requirements.range_policy = range;
    requirements.float_underflow = underflow;
    PcuDispatchKernelIr {
        numerical_requirements: requirements,
        id: PcuKernelId(97),
        entry: PcuDispatchEntryPoint {
            name: "portable-unary",
            logical_shape: [19, 1, 1],
        },
        bindings,
        ops,
        ports: &[],
        parameters: &[],
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}

fn assert_original_unary_headers(
    ir: &mut PcuDispatchKernelIr<'_>,
    portable: PcuCheckedFloatUnaryMapDescription,
) {
    ir.numerical_requirements.numerical_options.reproducibility = PcuReproducibility::Unspecified;
    let ordinary = describe_checked_float_unary_map(ir).unwrap();
    assert_eq!(ordinary.requirements, ir.numerical_requirements);
    let mut expected = portable;
    expected.requirements = ir.numerical_requirements;
    assert_eq!(ordinary, expected);
    assert_eq!(describe_portable_v1_unary_map(ir), Err(Error::NotRequested));
    ir.numerical_requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    assert_eq!(describe_checked_float_unary_map(ir), Ok(portable));
}

#[test]
fn six_formats_all_tuples_policies_and_geometries_keep_original_request() {
    let mut cases = 0;
    for scalar in FORMATS {
        let bindings = bindings(scalar);
        for operation in [Unary::Neg, Unary::Relu] {
            for range in [Range::Reject, Range::Clamp] {
                for underflow in POLICIES {
                    for grid in [false, true] {
                        for broadcast in [false, true] {
                            let index = if grid {
                                Index::GridStrideId
                            } else {
                                Index::InvocationId
                            };
                            let mut body = body(scalar, operation, index, range, underflow);
                            if broadcast {
                                let Op::Data(Data::BindingLoad { index, .. }) = &mut body[0] else {
                                    unreachable!()
                                };
                                *index = Index::BindingElementZero;
                            }
                            let outer = [
                                Op::GridStrideLoop {
                                    extent: 19,
                                    body: &body[..3],
                                },
                                Op::Control(PcuDispatchControlOp::Return),
                            ];
                            let mut ir = kernel(
                                &bindings,
                                if grid { &outer } else { &body },
                                range,
                                underflow,
                            );
                            if grid {
                                ir.entry.logical_shape[0] = 3;
                            }
                            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                                for compound in [
                                    PcuCompoundArithmeticPolicy::Checked,
                                    PcuCompoundArithmeticPolicy::BackendDefined,
                                ] {
                                    for precision in [
                                        PcuPrecisionPolicy::Preserve,
                                        PcuPrecisionPolicy::BackendOptimized,
                                    ] {
                                        ir.numerical_requirements.numerical_mode = mode;
                                        ir.numerical_requirements
                                            .numerical_options
                                            .compound_arithmetic = compound;
                                        ir.numerical_requirements.numerical_options.precision =
                                            precision;
                                        let actual = describe_portable_v1_unary_map(&ir).unwrap();
                                        assert_eq!(actual.scalar, scalar);
                                        assert_eq!(actual.operation, operation);
                                        assert_eq!(actual.requirements, ir.numerical_requirements);
                                        assert_eq!(actual.logical_extent, 19);
                                        assert_eq!(
                                            actual.submitted_invocations,
                                            if grid { 3 } else { 19 }
                                        );
                                        assert_eq!(actual.input_binding, INPUT);
                                        assert_eq!(actual.output_binding, OUTPUT);
                                        assert_eq!(actual.broadcast_input, broadcast);
                                        assert_eq!(
                                            actual.input_extent(),
                                            if broadcast { 1 } else { 19 }
                                        );
                                        assert_original_unary_headers(&mut ir, actual);
                                        cases += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 2304);
}

#[test]
fn request_header_mismatches_and_unsupported_formats_never_become_eligible() {
    for scalar in Scalar::ALL {
        let bindings = bindings(scalar);
        let body = body(
            scalar,
            Unary::Neg,
            Index::InvocationId,
            Range::Reject,
            Underflow::IeeeAfterRounding,
        );
        let ir = kernel(
            &bindings,
            &body,
            Range::Reject,
            Underflow::IeeeAfterRounding,
        );
        assert_eq!(
            describe_portable_v1_unary_map(&ir).is_ok(),
            FORMATS.contains(&scalar)
        );
    }
    let bindings = bindings(Scalar::F32);
    let body = body(
        Scalar::F32,
        Unary::Neg,
        Index::InvocationId,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    );
    let mut ir = kernel(
        &bindings,
        &body,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    );
    ir.numerical_requirements.numerical_options.reproducibility = PcuReproducibility::Unspecified;
    assert_eq!(
        describe_portable_v1_unary_map(&ir),
        Err(Error::NotRequested)
    );
    ir.numerical_requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    ir.numerical_requirements.range_policy = Range::Clamp;
    assert_eq!(
        describe_portable_v1_unary_map(&ir),
        Err(Error::RangeMismatch)
    );
    assert_eq!(
        describe_checked_float_unary_map(&ir),
        Err(PcuCheckedFloatUnaryMapError::RangeMismatch)
    );
    ir.numerical_requirements.range_policy = Range::Reject;
    ir.numerical_requirements.float_underflow = Underflow::AllowGradualUnderflow;
    assert_eq!(
        describe_portable_v1_unary_map(&ir),
        Err(Error::UnderflowMismatch)
    );
}

#[test]
fn access_and_original_declarations_are_validated_without_fake_inputs() {
    let bindings = bindings(Scalar::F32);
    let body = body(
        Scalar::F32,
        Unary::Relu,
        Index::BindingElementZero,
        Range::Clamp,
        Underflow::RejectSubnormalResult,
    );
    let ir = kernel(
        &bindings,
        &body,
        Range::Clamp,
        Underflow::RejectSubnormalResult,
    );
    assert_eq!(
        describe_portable_v1_unary_map(&ir).unwrap().input_extent(),
        1
    );
    let mut changed = bindings;
    changed[2].access = PcuBindingAccess::ReadWrite;
    assert_eq!(
        describe_portable_v1_unary_map(&kernel(
            &changed,
            &body,
            Range::Clamp,
            Underflow::RejectSubnormalResult
        )),
        Err(Error::InvalidAccess(INPUT))
    );
    for (slot, access) in [
        (2, PcuBindingAccess::WriteOnly),
        (1, PcuBindingAccess::ReadOnly),
    ] {
        let mut changed = bindings;
        changed[slot].access = access;
        assert!(matches!(
            describe_portable_v1_unary_map(&kernel(
                &changed,
                &body,
                Range::Clamp,
                Underflow::RejectSubnormalResult
            )),
            Err(Error::InvalidMap(_))
        ));
    }
    let mut changed = bindings;
    changed[0] = changed[2];
    assert!(matches!(
        describe_portable_v1_unary_map(&kernel(
            &changed,
            &body,
            Range::Clamp,
            Underflow::RejectSubnormalResult
        )),
        Err(Error::InvalidMap(_))
    ));
}

#[test]
fn valid_ssa_cannot_publish_the_input_instead_of_the_selected_result() {
    let bindings = bindings(Scalar::F64);
    let mut ops = body(
        Scalar::F64,
        Unary::Neg,
        Index::InvocationId,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    );
    let Op::Data(Data::BindingStore { value, .. }) = &mut ops[2] else {
        unreachable!()
    };
    *value = Id(0);
    let ir = kernel(&bindings, &ops, Range::Reject, Underflow::IeeeAfterRounding);
    crate::validate_typed_dispatch_value_flow(&ir).unwrap();
    assert_eq!(
        describe_portable_v1_unary_map(&ir),
        Err(Error::UnsupportedStructure)
    );
}

#[test]
fn logical_geometry_indices_and_extra_effects_reject() {
    let bindings = bindings(Scalar::F32);
    let body = body(
        Scalar::F32,
        Unary::Neg,
        Index::InvocationId,
        Range::Reject,
        Underflow::IeeeAfterRounding,
    );
    for shape in [[0, 1, 1], [19, 0, 1], [19, 2, 1], [19, 1, 2]] {
        let mut ir = kernel(
            &bindings,
            &body,
            Range::Reject,
            Underflow::IeeeAfterRounding,
        );
        ir.entry.logical_shape = shape;
        assert_eq!(
            describe_portable_v1_unary_map(&ir),
            Err(Error::InvalidLogicalShape(shape))
        );
    }
    let grid = [
        Op::GridStrideLoop {
            extent: 0,
            body: &body[..3],
        },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert!(
        describe_portable_v1_unary_map(&kernel(
            &bindings,
            &grid,
            Range::Reject,
            Underflow::IeeeAfterRounding
        ))
        .is_err()
    );
    let mut changed = body;
    let Op::Data(Data::BindingStore { index, .. }) = &mut changed[2] else {
        unreachable!()
    };
    *index = Index::BindingElementZero;
    assert!(matches!(
        describe_portable_v1_unary_map(&kernel(
            &bindings,
            &changed,
            Range::Reject,
            Underflow::IeeeAfterRounding
        )),
        Err(Error::InvalidMap(_))
    ));
    assert!(
        describe_portable_v1_unary_map(&kernel(
            &bindings,
            &body[..3],
            Range::Reject,
            Underflow::IeeeAfterRounding
        ))
        .is_err()
    );
    let mut changed = body;
    changed[0] = Op::Control(PcuDispatchControlOp::Return);
    assert_eq!(
        describe_portable_v1_unary_map(&kernel(
            &bindings,
            &changed,
            Range::Reject,
            Underflow::IeeeAfterRounding
        )),
        Err(Error::UnsupportedStructure)
    );
}

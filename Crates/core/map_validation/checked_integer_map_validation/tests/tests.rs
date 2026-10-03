use super::*;
#[rustfmt::skip]
use crate::{
    CheckedScalarMapResourceError as ResourceError,
    PcuBinding,
    PcuDispatchEntryPoint,
    PcuDispatchIntegerBinaryOp as Binary,
    PcuDispatchValueId as Id,
    PcuKernelId,
    PcuRangePolicy,
    PcuReproducibility,
};

const INPUT: PcuBindingRef = PcuBindingRef::new(4, 7);
const SEED: PcuBindingRef = PcuBindingRef::new(3, 2);
const STAGE: PcuBindingRef = PcuBindingRef::new(1, 3);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(8, 2);
const UNUSED: PcuBindingRef = PcuBindingRef::new(9, 11);
const INTEGERS: [PcuScalarType; 14] = [
    PcuScalarType::U8,
    PcuScalarType::U16,
    PcuScalarType::U32,
    PcuScalarType::U64,
    PcuScalarType::U128,
    PcuScalarType::U256,
    PcuScalarType::U512,
    PcuScalarType::I8,
    PcuScalarType::I16,
    PcuScalarType::I32,
    PcuScalarType::I64,
    PcuScalarType::I128,
    PcuScalarType::I256,
    PcuScalarType::I512,
];

fn declarations(scalar: PcuScalarType) -> [PcuBinding<'static>; 5] {
    [
        (UNUSED, PcuBindingAccess::ReadWrite),
        (OUTPUT, PcuBindingAccess::WriteOnly),
        (STAGE, PcuBindingAccess::ReadWrite),
        (INPUT, PcuBindingAccess::ReadOnly),
        (SEED, PcuBindingAccess::ReadOnly),
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

fn body(scalar: PcuScalarType, index: PcuDispatchIndex) -> [PcuDispatchOp<'static>; 9] {
    use PcuDispatchDataOp as Data;
    use PcuDispatchOp as Op;
    let binary = |op, result, lhs, rhs, range_policy| {
        Op::Data(Data::CheckedIntegerBinary {
            value_type: PcuValueType::Scalar(scalar),
            op,
            result: Id(result),
            lhs: Id(lhs),
            rhs: Id(rhs),
            range_policy,
        })
    };
    [
        Op::Data(Data::BindingLoad {
            result: Id(0),
            binding: INPUT,
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: SEED,
            index: PcuDispatchIndex::BindingElementZero,
        }),
        Op::Data(Data::BindingStore {
            binding: STAGE,
            index,
            value: Id(0),
        }),
        Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: STAGE,
            index,
        }),
        binary(Binary::Add, 3, 2, 1, PcuRangePolicy::Clamp),
        binary(Binary::Mul, 4, 3, 0, PcuRangePolicy::Reject),
        binary(Binary::Sub, 5, 4, 0, PcuRangePolicy::Clamp),
        Op::Data(Data::BindingStore {
            binding: OUTPUT,
            index,
            value: Id(5),
        }),
        Op::Control(PcuDispatchControlOp::Return),
    ]
}

fn kernel<'a>(
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [PcuDispatchOp<'a>],
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        id: PcuKernelId(315),
        entry: PcuDispatchEntryPoint {
            name: "checked-integer-composition",
            logical_shape: [23, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
    }
}

#[test]
fn fourteen_integer_profiles_share_exact_spans_initial_contents_and_original_headers() {
    for scalar in INTEGERS {
        for grid in [false, true] {
            let bindings = declarations(scalar);
            let data = body(
                scalar,
                if grid {
                    PcuDispatchIndex::GridStrideId
                } else {
                    PcuDispatchIndex::InvocationId
                },
            );
            let wrapped = [
                PcuDispatchOp::GridStrideLoop {
                    extent: 23,
                    body: &data[..8],
                },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let mut ir = kernel(&bindings, if grid { &wrapped } else { &data });
            if grid {
                ir.entry.logical_shape[0] = 3;
            }
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for reproducibility in [
                    PcuReproducibility::Unspecified,
                    PcuReproducibility::PortableV1,
                ] {
                    ir.numerical_requirements.range_policy = range;
                    ir.numerical_requirements.numerical_options.reproducibility = reproducibility;
                    let schema = assess_checked_integer_map_resources::<4>(
                        &ir,
                        PcuValueType::Scalar(scalar),
                        PcuValueTypeCaps::for_scalar(scalar),
                    )
                    .unwrap();
                    assert_eq!(schema.requirements, ir.numerical_requirements);
                    assert_eq!(schema.logical_extent, 23);
                    assert_eq!(schema.submitted_invocations, if grid { 3 } else { 23 });
                    assert_eq!(
                        schema
                            .resources()
                            .iter()
                            .map(|resource| resource.binding)
                            .collect::<std::vec::Vec<_>>(),
                        [INPUT, SEED, STAGE, OUTPUT]
                    );
                    assert!(schema.resource(UNUSED).is_none());
                    assert_eq!(
                        schema
                            .resource(INPUT)
                            .unwrap()
                            .minimum_initial_read_elements,
                        23
                    );
                    assert_eq!(
                        schema.resource(SEED).unwrap().minimum_initial_read_elements,
                        1
                    );
                    let stage = schema.resource(STAGE).unwrap();
                    assert_eq!(stage.minimum_read_elements, 23);
                    assert_eq!(stage.minimum_write_elements, 23);
                    assert_eq!(stage.minimum_initial_read_elements, 0);
                    assert!(!stage.has_cross_index_read_write());
                    assert_eq!(schema.resource(OUTPUT).unwrap().minimum_read_elements, 0);
                    assert_eq!(schema.resource(OUTPUT).unwrap().minimum_write_elements, 23);
                }
            }
        }
    }
}

#[test]
fn caller_capacity_and_cross_index_snapshot_obligations_are_separate() {
    let bindings = declarations(PcuScalarType::U512);
    let mut data = body(PcuScalarType::U512, PcuDispatchIndex::InvocationId);
    let ir = kernel(&bindings, &data);
    assert_eq!(
        assess_checked_integer_map_resources::<3>(
            &ir,
            PcuValueType::Scalar(PcuScalarType::U512),
            PcuValueTypeCaps::UINT512
        ),
        Err(ResourceError::InsufficientCapacity {
            capacity: 3,
            required_at_least: 4
        })
    );
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { index, .. }) = &mut data[3] else {
        unreachable!()
    };
    *index = PcuDispatchIndex::BindingElementZero;
    let ir = kernel(&bindings, &data);
    let schema = assess_checked_integer_map_resources::<4>(
        &ir,
        PcuValueType::Scalar(PcuScalarType::U512),
        PcuValueTypeCaps::UINT512,
    )
    .unwrap();
    let stage = schema.resource(STAGE).unwrap();
    assert!(stage.has_cross_index_read_write());
    assert_eq!(stage.minimum_initial_read_elements, 1);
    assert_eq!(stage.minimum_elements(), 23);
}

#[test]
fn unused_declarations_and_old_ssa_redefinitions_are_still_validated() {
    let mut bindings = declarations(PcuScalarType::U32);
    let mut data = body(PcuScalarType::U32, PcuDispatchIndex::InvocationId);
    bindings[0].binding_type = PcuBindingType::Value(PcuValueType::f32());
    assert_eq!(
        validate_checked_integer_map_kernel(
            &kernel(&bindings, &data),
            PcuValueType::u32(),
            PcuValueTypeCaps::UINT32
        ),
        Err(CheckedIntegerMapValidationError::UnsupportedRequirements)
    );
    bindings[0].binding_type = PcuBindingType::Value(PcuValueType::u32());
    bindings[0].storage = PcuBindingStorageClass::Uniform;
    assert_eq!(
        validate_checked_integer_map_kernel(
            &kernel(&bindings, &data),
            PcuValueType::u32(),
            PcuValueTypeCaps::UINT32,
        ),
        Err(CheckedIntegerMapValidationError::InvalidBinding(UNUSED))
    );
    bindings[0].storage = PcuBindingStorageClass::Storage;
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { result, .. }) = &mut data[4]
    else {
        unreachable!()
    };
    *result = Id(0);
    assert_eq!(
        validate_checked_integer_map_kernel(
            &kernel(&bindings, &data),
            PcuValueType::u32(),
            PcuValueTypeCaps::UINT32
        ),
        Err(CheckedIntegerMapValidationError::InvalidSsa)
    );
}

#[test]
fn integer_composition_never_widens_float_or_joint_division_profiles() {
    let bindings = declarations(PcuScalarType::U32);
    let mut data = body(PcuScalarType::U32, PcuDispatchIndex::InvocationId);
    assert_eq!(
        crate::validate_checked_float_map_kernel(
            &kernel(&bindings, &data),
            PcuValueType::u32(),
            PcuValueTypeCaps::UINT32
        ),
        Err(crate::CheckedFloatMapValidationError::UnsupportedType)
    );
    data[4] = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
        value_type: PcuValueType::u32(),
        flags: crate::model::PcuIntegerDivFlags::CHECKED,
        quotient: Id(3),
        remainder: Id(6),
        lhs: Id(2),
        rhs: Id(1),
    });
    assert_eq!(
        validate_checked_integer_map_kernel(
            &kernel(&bindings, &data),
            PcuValueType::u32(),
            PcuValueTypeCaps::UINT32
        ),
        Err(CheckedIntegerMapValidationError::UnsupportedOperation(4))
    );
}

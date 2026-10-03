//! Integer instruction admission must reject equally typed non-integer SSA.
#[rustfmt::skip]
use crate::{
    validate_integer_checked_div_rem_kernel,
    validate_typed_dispatch_value_flow,
    IntegerMapValidationError,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuKernelId,
    PcuRangePolicy,
    PcuScalarType,
    PcuTypedDispatchValidationError,
    PcuValueType,
    PcuValueTypeCaps,
    model::PcuIntegerDivFlags,
};

const INTEGERS: [PcuScalarType; 14] = [
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
];

fn bindings(value_type: PcuValueType) -> [PcuBinding<'static>; 4] {
    core::array::from_fn(|slot| {
        PcuBinding::value(
            None,
            0,
            u32::try_from(slot).unwrap(),
            PcuBindingStorageClass::Storage,
            if slot < 2 {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::WriteOnly
            },
            value_type,
        )
    })
}

fn body(value_type: PcuValueType, index: PcuDispatchIndex) -> [PcuDispatchOp<'static>; 5] {
    [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: PcuBindingRef::new(0, 1),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type,
            flags: PcuIntegerDivFlags::CHECKED,
            quotient: PcuDispatchValueId(3),
            remainder: PcuDispatchValueId(4),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 3),
            index,
            value: PcuDispatchValueId(4),
        }),
    ]
}

fn kernel<'a>(
    bindings: &'a [PcuBinding<'a>],
    ops: &'a [PcuDispatchOp<'a>],
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "integer_div_rem",
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

fn for_each_geometry(value_type: PcuValueType, mut check: impl FnMut(&PcuDispatchKernelIr<'_>)) {
    let bindings = bindings(value_type);
    let direct_body = body(value_type, PcuDispatchIndex::InvocationId);
    let direct = [
        direct_body[0],
        direct_body[1],
        direct_body[2],
        direct_body[3],
        direct_body[4],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    check(&kernel(&bindings, &direct));
    let grid_body = body(value_type, PcuDispatchIndex::GridStrideId);
    let grid = [
        PcuDispatchOp::GridStrideLoop {
            extent: 17,
            body: &grid_body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    check(&kernel(&bindings, &grid));
}

#[test]
fn all_fourteen_integer_types_keep_direct_and_grid_admission() {
    for scalar in INTEGERS {
        let value_type = PcuValueType::Scalar(scalar);
        for_each_geometry(value_type, |ir| {
            assert_eq!(validate_typed_dispatch_value_flow(ir), Ok(()));
            assert_eq!(
                validate_integer_checked_div_rem_kernel(
                    ir,
                    value_type,
                    PcuValueTypeCaps::for_scalar(scalar),
                ),
                Ok(()),
            );
        });
    }
}

#[test]
fn clamp_header_does_not_invent_checked_division_recovery() {
    for scalar in INTEGERS {
        let value_type = PcuValueType::Scalar(scalar);
        for_each_geometry(value_type, |ir| {
            // The instruction remains CHECKED. A broader header cannot invent
            // a useful clamped quotient/remainder for zero or signed MIN/-1.
            let mut ir = *ir;
            ir.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
            assert_eq!(
                validate_integer_checked_div_rem_kernel(
                    &ir,
                    value_type,
                    PcuValueTypeCaps::for_scalar(scalar),
                ),
                Err(IntegerMapValidationError::UnsupportedRequirements),
                "{scalar:?} must reject a division Clamp header",
            );
        });
    }
}

#[test]
fn matching_non_integer_scalar_types_do_not_define_integer_division() {
    let mut typed_rejected = 0;
    let mut structural_rejected = 0;
    for scalar in PcuScalarType::ALL {
        if INTEGERS.contains(&scalar) {
            continue;
        }
        let value_type = PcuValueType::Scalar(scalar);
        for_each_geometry(value_type, |ir| {
            typed_rejected += usize::from(
                validate_typed_dispatch_value_flow(ir)
                    == Err(PcuTypedDispatchValidationError::UnsupportedOperation(2)),
            );
            structural_rejected += usize::from(
                validate_integer_checked_div_rem_kernel(
                    ir,
                    value_type,
                    PcuValueTypeCaps::for_scalar(scalar),
                ) == Err(IntegerMapValidationError::UnsupportedInterface),
            );
        });
    }
    assert_eq!((typed_rejected, structural_rejected), (22, 22));
}

#[test]
fn scalar_integer_caps_do_not_admit_vector_or_matrix_division() {
    for value_type in [
        PcuValueType::Vector {
            scalar: PcuScalarType::I32,
            lanes: 2,
        },
        PcuValueType::Matrix {
            scalar: PcuScalarType::U64,
            rows: 2,
            cols: 2,
        },
    ] {
        for_each_geometry(value_type, |ir| {
            assert!(validate_typed_dispatch_value_flow(ir).is_err());
            assert_eq!(
                validate_integer_checked_div_rem_kernel(
                    ir,
                    value_type,
                    PcuValueTypeCaps::for_value_type(value_type),
                ),
                Err(IntegerMapValidationError::UnsupportedInterface),
            );
        });
    }
}

#[test]
fn reserved_total_division_flag_is_not_an_executable_checked_contract() {
    let value_type = PcuValueType::i32();
    let bindings = bindings(value_type);
    let mut ops = body(value_type, PcuDispatchIndex::InvocationId);
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { ref mut flags, .. }) = ops[2] else {
        unreachable!()
    };
    *flags = PcuIntegerDivFlags::DIV_OR_ZERO;
    let direct = [
        ops[0],
        ops[1],
        ops[2],
        ops[3],
        ops[4],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let ir = kernel(&bindings, &direct);
    assert_eq!(
        validate_typed_dispatch_value_flow(&ir),
        Err(PcuTypedDispatchValidationError::UnsupportedOperation(2)),
    );
    assert_eq!(
        validate_integer_checked_div_rem_kernel(&ir, value_type, PcuValueTypeCaps::INT32),
        Err(IntegerMapValidationError::UnsupportedOperation(2)),
    );
}

#[rustfmt::skip]
use super::{
    validate_integer_checked_binary_kernel,
    IntegerMapValidationError as Error,
};
#[rustfmt::skip]
use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp as BinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuKernelId,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};

#[allow(clippy::too_many_arguments)] // Fixture dimensions let tests vary index and SSA failures independently.
fn exercise(
    scalar: PcuScalarType,
    op: BinaryOp,
    lhs_index: PcuDispatchIndex,
    rhs_index: PcuDispatchIndex,
    result_id: Id,
    checked_lhs: Id,
    scalar_caps: PcuValueTypeCaps,
    grid: bool,
) -> Result<(), Error> {
    let value_type = PcuValueType::Scalar(scalar);
    let bindings = [
        PcuBinding::value(
            Some("lhs"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
        PcuBinding::value(
            Some("rhs"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
        PcuBinding::value(
            Some("out"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            value_type,
        ),
    ];
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 0),
            index: lhs_index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: PcuBindingRef::new(0, 1),
            index: rhs_index,
        }),
        Op::Data(Data::CheckedIntegerBinary {
            value_type,
            op,
            result: result_id,
            lhs: checked_lhs,
            rhs: Id(2),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: result_id,
        }),
    ];
    let loop_ops = [
        Op::GridStrideLoop {
            extent: 17,
            body: &body,
        },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let direct_ops = [
        body[0],
        body[1],
        body[2],
        body[3],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        id: PcuKernelId(7),
        entry: PcuDispatchEntryPoint {
            name: "checked-map",
            logical_shape: [17, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &loop_ops } else { &direct_ops },
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: crate::PcuDispatchFeatureCaps::empty(),
    };
    validate_integer_checked_binary_kernel(&kernel, value_type, op, scalar_caps)
}

#[test]
fn admits_all_eight_integer_widths_in_direct_and_grid_stride_maps() {
    for (scalar, caps) in [
        (PcuScalarType::I8, PcuValueTypeCaps::INT8),
        (PcuScalarType::U8, PcuValueTypeCaps::UINT8),
        (PcuScalarType::I16, PcuValueTypeCaps::INT16),
        (PcuScalarType::U16, PcuValueTypeCaps::UINT16),
        (PcuScalarType::I32, PcuValueTypeCaps::INT32),
        (PcuScalarType::U32, PcuValueTypeCaps::UINT32),
        (PcuScalarType::I64, PcuValueTypeCaps::INT64),
        (PcuScalarType::U64, PcuValueTypeCaps::UINT64),
    ] {
        assert_eq!(
            exercise(
                scalar,
                BinaryOp::Add,
                PcuDispatchIndex::InvocationId,
                PcuDispatchIndex::InvocationId,
                Id(3),
                Id(1),
                caps,
                false
            ),
            Ok(())
        );
        assert_eq!(
            exercise(
                scalar,
                BinaryOp::Mul,
                PcuDispatchIndex::GridStrideId,
                PcuDispatchIndex::GridStrideId,
                Id(3),
                Id(1),
                caps,
                true
            ),
            Ok(())
        );
    }
}

#[test]
fn admits_broadcast_and_rejects_invalid_type_ssa_and_capabilities() {
    assert_eq!(
        exercise(
            PcuScalarType::I32,
            BinaryOp::Sub,
            PcuDispatchIndex::BindingElementZero,
            PcuDispatchIndex::InvocationId,
            Id(3),
            Id(1),
            PcuValueTypeCaps::INT32,
            false
        ),
        Ok(())
    );
    assert_eq!(
        exercise(
            PcuScalarType::F32,
            BinaryOp::Add,
            PcuDispatchIndex::InvocationId,
            PcuDispatchIndex::InvocationId,
            Id(3),
            Id(1),
            PcuValueTypeCaps::FLOAT32,
            false
        ),
        Err(Error::UnsupportedInterface)
    );
    assert_eq!(
        exercise(
            PcuScalarType::U32,
            BinaryOp::Add,
            PcuDispatchIndex::InvocationId,
            PcuDispatchIndex::InvocationId,
            Id(3),
            Id(1),
            PcuValueTypeCaps::INT32,
            false
        ),
        Err(Error::UnsupportedRequirements)
    );
    assert_eq!(
        exercise(
            PcuScalarType::U32,
            BinaryOp::Add,
            PcuDispatchIndex::InvocationId,
            PcuDispatchIndex::InvocationId,
            Id(1),
            Id(1),
            PcuValueTypeCaps::UINT32,
            false
        ),
        Err(Error::InvalidValue(Id(1)))
    );
    assert_eq!(
        exercise(
            PcuScalarType::U32,
            BinaryOp::Add,
            PcuDispatchIndex::InvocationId,
            PcuDispatchIndex::InvocationId,
            Id(3),
            Id(9),
            PcuValueTypeCaps::UINT32,
            false
        ),
        Err(Error::InvalidValue(Id(9)))
    );
}

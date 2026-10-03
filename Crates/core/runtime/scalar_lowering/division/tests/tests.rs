use super::*;
#[rustfmt::skip]
use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuDispatchOpCaps,
    PcuDispatchValueId,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
    validate_typed_dispatch_value_flow,
};
use crate::model::PcuDispatchKernelBuilder;

fn fixture<T: PcuCheckedIntegerDivision>() {
    let bindings = [
        PcuBinding::scalar::<T>(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            None,
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
        PcuBinding::scalar::<T>(
            None,
            0,
            3,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
    ];
    let builder =
        PcuDispatchKernelBuilder::<6>::new(1, "joint_helper", [7, 1, 1]).with_bindings(&bindings);
    let mut lowering = PcuScalarLowering::new(builder, 1);
    let lhs = lowering
        .load_value::<T>(PcuBindingRef::new(0, 0), PcuDispatchIndex::InvocationId)
        .unwrap();
    let rhs = lowering
        .load_value::<T>(
            PcuBindingRef::new(0, 1),
            PcuDispatchIndex::BindingElementZero,
        )
        .unwrap();
    let (quotient, remainder) = lowering.checked_div_rem_values(lhs, rhs).unwrap();
    assert_eq!(quotient.id(), PcuDispatchValueId(3));
    assert_eq!(remainder.id(), PcuDispatchValueId(4));
    // Publish remainder first: SSA identity is independent of destination order.
    lowering
        .store_value(
            PcuBindingRef::new(0, 2),
            PcuDispatchIndex::InvocationId,
            remainder,
        )
        .unwrap();
    lowering
        .store_value(
            PcuBindingRef::new(0, 3),
            PcuDispatchIndex::InvocationId,
            quotient,
        )
        .unwrap();
    let builder = lowering
        .finish()
        .unwrap()
        .with_control_op(PcuDispatchControlOp::Return)
        .unwrap();
    builder.with_ir(|ir| {
        assert_eq!(validate_typed_dispatch_value_flow(ir), Ok(()));
        assert_eq!(
            ir.ops[2],
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::Scalar(T::TYPE),
                flags: PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            })
        );
        assert!(
            ir.required_instruction_support()
                .contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        );
        assert!(
            ir.required_type_support()
                .contains(crate::PcuValueTypeCaps::for_scalar(T::TYPE))
        );
    });
}

#[test]
fn typed_joint_lowering_preserves_all_fourteen_widths_and_both_results() {
    fixture::<u8>();
    fixture::<i8>();
    fixture::<u16>();
    fixture::<i16>();
    fixture::<u32>();
    fixture::<i32>();
    fixture::<u64>();
    fixture::<i64>();
    fixture::<u128>();
    fixture::<i128>();
    fixture::<PcuU256>();
    fixture::<PcuI256>();
    fixture::<PcuU512>();
    fixture::<PcuI512>();
}

#[test]
fn clamp_refusal_does_not_reserve_values_or_emit_joint_arithmetic() {
    let builder = PcuDispatchKernelBuilder::<3>::new(1, "clamp_refusal", [1, 1, 1]);
    let mut lowering = PcuScalarLowering::new(builder, 3).with_range_policy(PcuRangePolicy::Clamp);
    let lhs = PcuScalarValue::<u32>::from_id(PcuDispatchValueId(1));
    let rhs = PcuScalarValue::<u32>::from_id(PcuDispatchValueId(2));
    assert!(
        matches!(lowering.checked_div_rem_values(lhs, rhs), Err(error) if error == PcuError::unsupported())
    );
    assert_eq!(lowering.next_value, 3);
    lowering = lowering.with_range_policy(PcuRangePolicy::Reject);
    let (quotient, remainder) = lowering.checked_div_rem_values(lhs, rhs).unwrap();
    assert_eq!(quotient.id(), PcuDispatchValueId(3));
    assert_eq!(remainder.id(), PcuDispatchValueId(4));
    lowering
        .finish()
        .unwrap()
        .with_ir(|ir| assert_eq!(ir.ops.len(), 1));
}

#[test]
fn both_ssa_values_and_one_instruction_must_fit_caller_storage() {
    let lhs = PcuScalarValue::<u32>::from_id(PcuDispatchValueId(1));
    let rhs = PcuScalarValue::<u32>::from_id(PcuDispatchValueId(2));
    for first in [u16::MAX - 1, u16::MAX] {
        let builder = PcuDispatchKernelBuilder::<1>::new(1, "exhausted_values", [1, 1, 1]);
        let mut lowering = PcuScalarLowering::new(builder, first);
        assert!(
            matches!(lowering.checked_div_rem_values(lhs, rhs), Err(error) if error == PcuError::resource_exhausted())
        );
    }
    let builder = PcuDispatchKernelBuilder::<0>::new(1, "exhausted_ops", [1, 1, 1]);
    let mut lowering = PcuScalarLowering::new(builder, 3);
    assert!(
        matches!(lowering.checked_div_rem_values(lhs, rhs), Err(error) if error == PcuError::resource_exhausted())
    );
    // Failed builder insertion is poisoned; incomplete caller output cannot escape.
    assert!(lowering.finish().is_err());
}

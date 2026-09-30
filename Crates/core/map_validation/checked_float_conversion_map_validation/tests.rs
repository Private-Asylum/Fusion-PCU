#[rustfmt::skip]
use super::{
    validate_checked_float_conversion_map_kernel,
    CheckedFloatConversionMapValidationError as Error,
};
#[rustfmt::skip]
use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchCheckedFloatConversion as Conversion,
    PcuDispatchControlOp,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp as Binary,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuFloatUnderflowPolicy as Policy,
    PcuKernelId,
    PcuValueType,
    PcuValueTypeCaps,
};

fn binding(slot: u32, value_type: PcuValueType, access: PcuBindingAccess) -> PcuBinding<'static> {
    PcuBinding::value(
        Some("data"),
        0,
        slot,
        PcuBindingStorageClass::Storage,
        access,
        value_type,
    )
}

fn kernel<'a>(bindings: &'a [PcuBinding<'a>], ops: &'a [Op<'a>]) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "checked-float-convert-map",
            logical_shape: [4, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}

fn required_float_caps() -> PcuValueTypeCaps {
    PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64
}

fn valid_bindings() -> [PcuBinding<'static>; 3] {
    [
        binding(0, PcuValueType::f64(), PcuBindingAccess::ReadOnly),
        binding(1, PcuValueType::f64(), PcuBindingAccess::ReadOnly),
        binding(2, PcuValueType::f32(), PcuBindingAccess::WriteOnly),
    ]
}

fn valid_body(index: PcuDispatchIndex) -> [Op<'static>; 8] {
    [
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::BindingElementZero,
        }),
        Op::Data(Data::CheckedFloatBinary {
            value_type: PcuValueType::f64(),
            op: Binary::Sub,
            underflow_policy: Policy::AllowGradualUnderflow,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(3),
            lhs: Id(1),
            rhs: Id(2),
        }),
        Op::Data(Data::CheckedFloatConvert {
            conversion: Conversion::F64ToF32,
            underflow_policy: Policy::IeeeAfterRounding,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(4),
            value: Id(3),
        }),
        Op::Data(Data::Constant {
            result: Id(5),
            value: crate::PcuParameterValue::F32(1.0_f32.to_bits()),
        }),
        Op::Data(Data::CheckedFloatBinary {
            value_type: PcuValueType::f32(),
            op: Binary::Mul,
            underflow_policy: Policy::RejectSubnormalResult,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(6),
            lhs: Id(4),
            rhs: Id(5),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index,
            value: Id(6),
        }),
        Op::Control(PcuDispatchControlOp::Return),
    ]
}

#[test]
fn admits_mixed_width_checked_map_with_conversion_and_arithmetic_on_both_sides() {
    let bindings = valid_bindings();
    let ops = valid_body(PcuDispatchIndex::InvocationId);
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &ops),
            required_float_caps(),
        ),
        Ok(())
    );
}

#[test]
fn admits_grid_stride_map_with_broadcast_inputs() {
    let bindings = valid_bindings();
    let body = valid_body(PcuDispatchIndex::GridStrideId);
    let ops = [
        Op::GridStrideLoop {
            extent: 64,
            body: &body[..body.len() - 1],
        },
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &ops),
            required_float_caps(),
        ),
        Ok(())
    );
}

#[test]
fn admits_literal_only_and_multiple_casts_without_any_load() {
    let bindings = [binding(0, PcuValueType::f32(), PcuBindingAccess::WriteOnly)];
    let ops = [
        Op::Data(Data::Constant {
            result: Id(1),
            value: crate::PcuParameterValue::F64(2.0_f64.to_bits()),
        }),
        Op::Data(Data::CheckedFloatConvert {
            conversion: Conversion::F64ToF32,
            underflow_policy: Policy::default(),
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(2),
            value: Id(1),
        }),
        Op::Data(Data::Constant {
            result: Id(3),
            value: crate::PcuParameterValue::F64(1.0_f64.to_bits()),
        }),
        Op::Data(Data::CheckedFloatConvert {
            conversion: Conversion::F64ToF32,
            underflow_policy: Policy::AllowGradualUnderflow,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(4),
            value: Id(3),
        }),
        Op::Data(Data::CheckedFloatBinary {
            value_type: PcuValueType::f32(),
            op: Binary::Add,
            underflow_policy: Policy::default(),
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(5),
            lhs: Id(2),
            rhs: Id(4),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
            value: Id(5),
        }),
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &ops),
            required_float_caps(),
        ),
        Ok(())
    );
}

#[test]
fn admits_read_only_f32_peer_and_read_write_output() {
    let bindings = [
        binding(0, PcuValueType::f64(), PcuBindingAccess::ReadOnly),
        binding(1, PcuValueType::f32(), PcuBindingAccess::ReadOnly),
        binding(2, PcuValueType::f32(), PcuBindingAccess::ReadWrite),
    ];
    let ops = [
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        Op::Data(Data::CheckedFloatConvert {
            conversion: Conversion::F64ToF32,
            underflow_policy: Policy::default(),
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(2),
            value: Id(1),
        }),
        Op::Data(Data::BindingLoad {
            result: Id(3),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::BindingElementZero,
        }),
        Op::Data(Data::BindingLoad {
            result: Id(4),
            binding: PcuBindingRef::new(0, 2),
            index: PcuDispatchIndex::InvocationId,
        }),
        Op::Data(Data::CheckedFloatBinary {
            value_type: PcuValueType::f32(),
            op: Binary::Add,
            underflow_policy: Policy::default(),
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(5),
            lhs: Id(2),
            rhs: Id(3),
        }),
        Op::Data(Data::CheckedFloatBinary {
            value_type: PcuValueType::f32(),
            op: Binary::Add,
            underflow_policy: Policy::default(),
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(6),
            lhs: Id(5),
            rhs: Id(4),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 2),
            index: PcuDispatchIndex::InvocationId,
            value: Id(6),
        }),
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &ops),
            required_float_caps(),
        ),
        Ok(())
    );
}

#[test]
fn admits_checked_f32_to_f64_widening_to_f64_output() {
    let bindings = [
        binding(0, PcuValueType::f32(), PcuBindingAccess::ReadOnly),
        binding(1, PcuValueType::f64(), PcuBindingAccess::WriteOnly),
    ];
    let ops = [
        Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        Op::Data(Data::CheckedFloatConvert {
            conversion: Conversion::F32ToF64,
            underflow_policy: Policy::RejectSubnormalResult,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(2),
            value: Id(1),
        }),
        Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
            value: Id(2),
        }),
        Op::Control(PcuDispatchControlOp::Return),
    ];
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &ops),
            required_float_caps(),
        ),
        Ok(())
    );
}

#[test]
fn rejects_missing_width_capability_and_missing_conversion() {
    let bindings = valid_bindings();
    let ops = valid_body(PcuDispatchIndex::InvocationId);
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &ops),
            PcuValueTypeCaps::FLOAT32,
        ),
        Err(Error::UnsupportedRequirements)
    );
    let no_cast = [ops[0], ops[1], ops[2], ops[4], ops[5], ops[6], ops[7]];
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &no_cast),
            required_float_caps(),
        ),
        Err(Error::MissingConversion)
    );
}

#[test]
fn rejects_bad_input_access_output_type_flow_and_indices() {
    let mut bindings = valid_bindings();
    bindings[0] = binding(0, PcuValueType::f64(), PcuBindingAccess::WriteOnly);
    let ops = valid_body(PcuDispatchIndex::InvocationId);
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &ops),
            required_float_caps(),
        ),
        Err(Error::InvalidBinding(PcuBindingRef::new(0, 0)))
    );

    bindings = valid_bindings();
    bindings[2] = binding(2, PcuValueType::f64(), PcuBindingAccess::WriteOnly);
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &ops),
            required_float_caps(),
        ),
        // Both widths are valid storage; a narrowed F32 value cannot be stored as F64.
        Err(Error::InvalidSsa)
    );

    bindings = valid_bindings();
    let mut bad_index = ops;
    bad_index[0] = Op::Data(Data::BindingLoad {
        result: Id(1),
        binding: PcuBindingRef::new(0, 0),
        index: PcuDispatchIndex::GridStrideId,
    });
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &bad_index),
            required_float_caps(),
        ),
        Err(Error::InvalidIndex(0))
    );
}

#[test]
fn rejects_invalid_ssa_and_missing_terminal_return() {
    let bindings = valid_bindings();
    let mut ops = valid_body(PcuDispatchIndex::InvocationId);
    ops[3] = Op::Data(Data::CheckedFloatConvert {
        conversion: Conversion::F64ToF32,
        underflow_policy: Policy::default(),
        range_policy: crate::PcuRangePolicy::Reject,
        result: Id(4),
        value: Id(99),
    });
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, &ops),
            required_float_caps(),
        ),
        Err(Error::InvalidSsa)
    );
    let missing_return = &ops[..ops.len() - 1];
    assert_eq!(
        validate_checked_float_conversion_map_kernel(
            &kernel(&bindings, missing_return),
            required_float_caps(),
        ),
        Err(Error::MissingReturn)
    );
}

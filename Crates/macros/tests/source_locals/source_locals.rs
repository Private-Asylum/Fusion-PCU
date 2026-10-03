//! Genuine source specialization and SSA effects; no native provider claim.
#[path = "compound/compound.rs"]
mod compound;
#[path = "helpers/helpers.rs"]
mod helpers;
#[path = "mutation/mutation.rs"]
mod mutation;
#[path = "transport/transport.rs"]
mod transport;
#[path = "typed/typed.rs"]
mod typed;

use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    assess_checked_float_map_resources,
    validate_typed_dispatch_value_flow,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuCheckedInteger,
    PcuDispatchDataOp as Data,
    PcuDispatchFloatBinaryOp as FloatOp,
    PcuDispatchIntegerBinaryOp as IntegerOp,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
};

#[pcu(invocations = N, crate_path = ::pcu_alias, flag(reject_subnormal_result), flag(clamp_range))]
fn float_locals<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let value = input[id];
    let doubled = value + value;
    output[id] = doubled * value;
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn grid_locals<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let value = input[id];
        let value = value + input[0];
        output[id] = value * value;
        id += stride;
    }
}

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn matrix_locals<T: PcuCheckedFloat, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    let row = id / C;
    let col = id % C;
    let value = input[row][col];
    let doubled = value + value;
    output[row][col] = doubled * value;
}

#[pcu(invocations = N, crate_path = ::pcu_alias, flag(clamp_range))]
fn integer_locals<T: PcuCheckedInteger, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let value = input[id];
    let doubled = value + value;
    output[id] = doubled * value;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn unused_checked_local<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    denominator: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let _must_check = input[id] / denominator[id];
    let value = input[id];
    output[id] = value + value;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn ordered_float<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    stage[id] = original + original;
    let updated = stage[id];
    output[id] = updated * original;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn ordered_integer<T: PcuCheckedInteger, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    output[id] = original + original;
    let updated = output[id];
    output[id] = updated * original;
}

fn verify_float<T: PcuCheckedFloat>() {
    let bindings = float_locals_bindings::<T>();
    let builder = float_locals_ir::<T, 7>(&bindings).unwrap();
    let ir = builder.ir();
    validate_typed_dispatch_value_flow(&ir).unwrap();
    assert!(matches!(
        ir.ops[0],
        Op::Data(Data::BindingLoad { result: Id(1), .. })
    ));
    assert!(matches!(ir.ops[1], Op::Data(Data::CheckedFloatBinary {
        op: FloatOp::Add, lhs: Id(1), rhs: Id(1), result: Id(2),
        value_type, range_policy: PcuRangePolicy::Clamp,
        underflow_policy: PcuFloatUnderflowPolicy::RejectSubnormalResult,
    }) if value_type == PcuValueType::Scalar(T::TYPE)));
    assert!(matches!(ir.ops[2], Op::Data(Data::CheckedFloatBinary {
        op: FloatOp::Mul, lhs: Id(2), rhs: Id(1), result: Id(3),
        value_type, range_policy: PcuRangePolicy::Clamp,
        underflow_policy: PcuFloatUnderflowPolicy::RejectSubnormalResult,
    }) if value_type == PcuValueType::Scalar(T::TYPE)));
    assert!(matches!(
        ir.ops[3],
        Op::Data(Data::BindingStore { value: Id(3), .. })
    ));

    let bindings = grid_locals_bindings::<T>();
    grid_locals_ir::<T, 19>(&bindings).unwrap().with_ir(|ir| {
        validate_typed_dispatch_value_flow(ir).unwrap();
        let schema = assess_checked_float_map_resources::<2>(
            ir,
            PcuValueType::Scalar(T::TYPE),
            PcuValueTypeCaps::for_scalar(T::TYPE),
        )
        .unwrap();
        assert_eq!(schema.submitted_invocations, 3);
        assert_eq!(schema.logical_extent, 19);
        let input = schema.resource(PcuBindingRef::new(0, 0)).unwrap();
        assert!(input.reads_element_zero);
        assert_eq!(input.minimum_read_elements, 19);
        assert!(!input.has_cross_index_read_write());
        let Op::GridStrideLoop { body, .. } = ir.ops[0] else {
            panic!("grid body")
        };
        assert!(matches!(
            body[3],
            Op::Data(Data::CheckedFloatBinary {
                op: FloatOp::Mul,
                lhs: Id(3),
                rhs: Id(3),
                ..
            })
        ));
    });

    let bindings = matrix_locals_bindings::<T>();
    let builder = matrix_locals_ir::<T, 2, 3>(&bindings).unwrap();
    let ir = builder.ir();
    validate_typed_dispatch_value_flow(&ir).unwrap();
    assert_eq!(ir.entry.logical_shape, [6, 1, 1]);
    assert_eq!(ir.ops.len(), 5);

    let bindings = ordered_float_bindings::<T>();
    let builder = ordered_float_ir::<T, 7>(&bindings).unwrap();
    let ir = builder.ir();
    validate_typed_dispatch_value_flow(&ir).unwrap();
    assert!(matches!(ir.ops[2], Op::Data(Data::BindingStore {
        binding, value: Id(2), ..
    }) if binding == PcuBindingRef::new(0, 1)));
    assert!(matches!(ir.ops[3], Op::Data(Data::BindingLoad {
        binding, result: Id(3), ..
    }) if binding == PcuBindingRef::new(0, 1)));
    assert!(matches!(
        ir.ops[4],
        Op::Data(Data::CheckedFloatBinary {
            lhs: Id(3),
            rhs: Id(1),
            result: Id(4),
            op: FloatOp::Mul,
            ..
        })
    ));
    assert!(matches!(ir.ops[5], Op::Data(Data::BindingStore {
        binding, value: Id(4), ..
    }) if binding == PcuBindingRef::new(0, 2)));
    let schema = assess_checked_float_map_resources::<3>(
        &ir,
        PcuValueType::Scalar(T::TYPE),
        PcuValueTypeCaps::for_scalar(T::TYPE),
    )
    .unwrap();
    let stage = schema.resource(PcuBindingRef::new(0, 1)).unwrap();
    assert_eq!(stage.minimum_read_elements, 7);
    assert_eq!(stage.minimum_initial_read_elements, 0);
    assert_eq!(stage.minimum_write_elements, 7);
    assert!(!stage.has_cross_index_read_write());
}

#[test]
fn six_format_typed_locals_keep_policy_order_and_shared_values() {
    verify_float::<PcuF16Bits>();
    verify_float::<PcuBf16Bits>();
    verify_float::<PcuF8E4M3FnBits>();
    verify_float::<PcuF8E5M2Bits>();
    verify_float::<f32>();
    verify_float::<f64>();
}

fn verify_integer<T: PcuCheckedInteger>() {
    let bindings = integer_locals_bindings::<T>();
    let builder = integer_locals_ir::<T, 7>(&bindings).unwrap();
    let ir = builder.ir();
    validate_typed_dispatch_value_flow(&ir).unwrap();
    assert!(matches!(ir.ops[1], Op::Data(Data::CheckedIntegerBinary {
        value_type, op: IntegerOp::Add, lhs: Id(1), rhs: Id(1), result: Id(2),
        range_policy: PcuRangePolicy::Clamp,
    }) if value_type == PcuValueType::Scalar(T::TYPE)));
    assert!(matches!(ir.ops[2], Op::Data(Data::CheckedIntegerBinary {
        value_type, op: IntegerOp::Mul, lhs: Id(2), rhs: Id(1), result: Id(3),
        range_policy: PcuRangePolicy::Clamp,
    }) if value_type == PcuValueType::Scalar(T::TYPE)));

    let bindings = ordered_integer_bindings::<T>();
    let builder = ordered_integer_ir::<T, 7>(&bindings).unwrap();
    let ir = builder.ir();
    validate_typed_dispatch_value_flow(&ir).unwrap();
    assert!(matches!(
        ir.ops[2],
        Op::Data(Data::BindingStore { value: Id(2), .. })
    ));
    assert!(matches!(
        ir.ops[3],
        Op::Data(Data::BindingLoad { result: Id(3), .. })
    ));
    assert!(matches!(ir.ops[4], Op::Data(Data::CheckedIntegerBinary {
        value_type, lhs: Id(3), rhs: Id(1), result: Id(4), op: IntegerOp::Mul, ..
    }) if value_type == PcuValueType::Scalar(T::TYPE)));
}

#[test]
fn fourteen_integer_widths_keep_checked_local_ssa_without_inline_constants() {
    verify_integer::<i8>();
    verify_integer::<u8>();
    verify_integer::<i16>();
    verify_integer::<u16>();
    verify_integer::<i32>();
    verify_integer::<u32>();
    verify_integer::<i64>();
    verify_integer::<u64>();
    verify_integer::<i128>();
    verify_integer::<u128>();
    verify_integer::<PcuI256>();
    verify_integer::<PcuU256>();
    verify_integer::<PcuI512>();
    verify_integer::<PcuU512>();
}

#[test]
fn unused_checked_value_retains_fault_order_and_denominator_resource() {
    let bindings = unused_checked_local_bindings::<f32>();
    let builder = unused_checked_local_ir::<f32, 7>(&bindings).unwrap();
    let ir = builder.ir();
    let schema = assess_checked_float_map_resources::<3>(
        &ir,
        PcuValueType::f32(),
        PcuValueTypeCaps::FLOAT32,
    )
    .unwrap();
    assert_eq!(schema.resources().len(), 3);
    assert_eq!(
        schema
            .resource(PcuBindingRef::new(0, 1))
            .unwrap()
            .minimum_read_elements,
        7
    );
    let arithmetic = ir
        .ops
        .iter()
        .filter_map(|op| match op {
            Op::Data(Data::CheckedFloatBinary { op, .. }) => Some(*op),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(arithmetic, [FloatOp::Div, FloatOp::Add]);
}

//! Actual generic source retains one unique input and each independent read index.
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    assess_checked_integer_binary_operands,
    validate_integer_checked_binary_kernel,
    validate_typed_dispatch_value_flow,
    PcuBindingRef,
    PcuCheckedInteger,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
    PcuValueType,
    PcuValueTypeCaps,
};

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn square<T: PcuCheckedInteger, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}

#[pcu(invocations = 3, flag(clamp_range), crate_path = ::pcu_alias)]
fn broadcast_difference<T: PcuCheckedInteger, const N: usize>(output: &mut [T], input: &[T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[0] - input[id];
        id += stride;
    }
}

fn assert_source_roles<T: PcuCheckedInteger>() {
    let bindings = square_bindings::<T>();
    let builder = square_ir::<T, 19>(&bindings).unwrap();
    let ir = builder.ir();
    validate_typed_dispatch_value_flow(&ir).unwrap();
    let schema = assess_checked_integer_binary_operands(
        &ir,
        PcuValueType::Scalar(T::TYPE),
        PcuDispatchIntegerBinaryOp::Mul,
        PcuValueTypeCaps::for_scalar(T::TYPE),
    )
    .unwrap();
    assert_eq!(schema.input_bindings(), &[PcuBindingRef::new(0, 2)]);
    assert_eq!(schema.output_binding(), PcuBindingRef::new(0, 1));
    assert_eq!(schema.operand_inputs(), [0, 0]);
    assert_eq!(schema.input_element_counts(19), [19, 0]);
    assert!(
        validate_integer_checked_binary_kernel(
            &ir,
            PcuValueType::Scalar(T::TYPE),
            PcuDispatchIntegerBinaryOp::Mul,
            PcuValueTypeCaps::for_scalar(T::TYPE),
        )
        .is_err()
    );

    let bindings = broadcast_difference_bindings::<T>();
    let builder = broadcast_difference_ir::<T, 19>(&bindings).unwrap();
    builder.with_ir(|ir| {
        validate_typed_dispatch_value_flow(ir).unwrap();
        let schema = assess_checked_integer_binary_operands(
            ir,
            PcuValueType::Scalar(T::TYPE),
            PcuDispatchIntegerBinaryOp::Sub,
            PcuValueTypeCaps::for_scalar(T::TYPE),
        )
        .unwrap();
        assert_eq!(schema.input_bindings(), &[PcuBindingRef::new(0, 1)]);
        assert_eq!(schema.output_binding(), PcuBindingRef::new(0, 0));
        assert_eq!(schema.operand_inputs(), [0, 0]);
        assert_eq!(
            schema.operand_indices(),
            [
                PcuDispatchIndex::BindingElementZero,
                PcuDispatchIndex::GridStrideId
            ]
        );
        assert_eq!(schema.input_element_counts(19), [19, 0]);
        assert_eq!(
            ir.numerical_requirements.range_policy,
            pcu_alias::PcuRangePolicy::Clamp
        );
    });
}

#[test]
fn all_fourteen_generic_integer_sources_preserve_actual_operand_roles() {
    assert_source_roles::<u8>();
    assert_source_roles::<i8>();
    assert_source_roles::<u16>();
    assert_source_roles::<i16>();
    assert_source_roles::<u32>();
    assert_source_roles::<i32>();
    assert_source_roles::<u64>();
    assert_source_roles::<i64>();
    assert_source_roles::<u128>();
    assert_source_roles::<i128>();
    assert_source_roles::<PcuU256>();
    assert_source_roles::<PcuI256>();
    assert_source_roles::<PcuU512>();
    assert_source_roles::<PcuI512>();
}

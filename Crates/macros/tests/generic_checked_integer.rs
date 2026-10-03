//! Generic source arithmetic retains exact integer representation and checked range contracts.
#[path = "generic_checked_integer/operand_schema/operand_schema.rs"]
mod operand_schema;
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    validate_integer_checked_binary_kernel,
    validate_typed_dispatch_value_flow,
    PcuCheckedInteger,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchOp,
    PcuRangePolicy,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
    PcuValueType,
    PcuValueTypeCaps,
};

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn sum<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn difference<T, const N: usize>(left: &[T], right: &[T], output: &mut [T])
where
    T: PcuCheckedInteger,
{
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id] - right[id];
        id += stride;
    }
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn scale<T: PcuCheckedInteger, const N: usize>(input: &[T], factor: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] * *factor;
}

#[pcu(invocations = N, flag(clamp_range), crate_path = ::pcu_alias)]
fn clamp_sum<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}

#[pcu(invocations = 3, flag(clamp_range), crate_path = ::pcu_alias)]
fn clamp_scale<T: PcuCheckedInteger, const N: usize>(input: &[T], factor: &T, output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[id] * *factor;
        id += stride;
    }
}

fn check_clamp_ir<T: PcuCheckedInteger>() {
    let bindings = clamp_sum_bindings::<T>();
    let builder = clamp_sum_ir::<T, 7>(&bindings).unwrap();
    let direct = builder.ir();
    let bindings = clamp_scale_bindings::<T>();
    let builder = clamp_scale_ir::<T, 7>(&bindings).unwrap();
    builder.with_ir(|grid| {
        for (kernel, operation) in [
            (&direct, PcuDispatchIntegerBinaryOp::Add),
            (grid, PcuDispatchIntegerBinaryOp::Mul),
        ] {
            assert_eq!(
                kernel.numerical_requirements.range_policy,
                PcuRangePolicy::Clamp
            );
            assert!(
                kernel
                    .required_feature_support()
                    .contains(PcuDispatchFeatureCaps::RANGE_CLAMP)
            );
            validate_typed_dispatch_value_flow(kernel).unwrap();
            validate_integer_checked_binary_kernel(
                kernel,
                PcuValueType::Scalar(T::TYPE),
                operation,
                PcuValueTypeCaps::for_scalar(T::TYPE),
            )
            .unwrap();
            let ops = match kernel.ops.first() {
                Some(PcuDispatchOp::GridStrideLoop { body, .. }) => *body,
                _ => kernel.ops,
            };
            assert!(ops.iter().any(|op| matches!(
                op,
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                    range_policy: PcuRangePolicy::Clamp,
                    ..
                })
            )));
        }
    });
}

#[test]
fn all_fourteen_integer_formats_carry_clamp_through_direct_and_grid_broadcast_source() {
    check_clamp_ir::<u8>();
    check_clamp_ir::<i8>();
    check_clamp_ir::<u16>();
    check_clamp_ir::<i16>();
    check_clamp_ir::<u32>();
    check_clamp_ir::<i32>();
    check_clamp_ir::<u64>();
    check_clamp_ir::<i64>();
    check_clamp_ir::<u128>();
    check_clamp_ir::<i128>();
    check_clamp_ir::<PcuU256>();
    check_clamp_ir::<PcuI256>();
    check_clamp_ir::<PcuU512>();
    check_clamp_ir::<PcuI512>();
}

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn matrix_sum<T: PcuCheckedInteger, const R: usize, const C: usize>(
    left: &[[T; C]; R],
    right: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = context.global_invocation_id;
    output[id / C][id % C] = left[id / C][id % C] + right[id / C][id % C];
}

fn check_ir<T: PcuCheckedInteger>() {
    let bindings = sum_bindings::<T>();
    let builder = sum_ir::<T, 7>(&bindings).unwrap();
    let kernel = builder.ir();
    validate_typed_dispatch_value_flow(&kernel).unwrap();
    validate_integer_checked_binary_kernel(
        &kernel,
        PcuValueType::Scalar(T::TYPE),
        PcuDispatchIntegerBinaryOp::Add,
        PcuValueTypeCaps::for_scalar(T::TYPE),
    )
    .unwrap();
    let bindings = difference_bindings::<T>();
    let builder = difference_ir::<T, 7>(&bindings).unwrap();
    builder.with_ir(|kernel| {
        validate_typed_dispatch_value_flow(kernel).unwrap();
        validate_integer_checked_binary_kernel(
            kernel,
            PcuValueType::Scalar(T::TYPE),
            PcuDispatchIntegerBinaryOp::Sub,
            PcuValueTypeCaps::for_scalar(T::TYPE),
        )
        .unwrap();
    });
    let bindings = scale_bindings::<T>();
    let builder = scale_ir::<T, 7>(&bindings).unwrap();
    let kernel = builder.ir();
    assert_eq!(
        kernel.minimum_binding_elements_for(pcu_alias::PcuBindingRef::new(0, 1), 7),
        1
    );
    validate_integer_checked_binary_kernel(
        &kernel,
        PcuValueType::Scalar(T::TYPE),
        PcuDispatchIntegerBinaryOp::Mul,
        PcuValueTypeCaps::for_scalar(T::TYPE),
    )
    .unwrap();
    let bindings = matrix_sum_bindings::<T>();
    let builder = matrix_sum_ir::<T, 2, 3>(&bindings).unwrap();
    validate_typed_dispatch_value_flow(&builder.ir()).unwrap();
}

#[test]
fn all_fourteen_integer_representations_specialize_exact_checked_ir() {
    check_ir::<u8>();
    check_ir::<i8>();
    check_ir::<u16>();
    check_ir::<i16>();
    check_ir::<u32>();
    check_ir::<i32>();
    check_ir::<u64>();
    check_ir::<i64>();
    check_ir::<u128>();
    check_ir::<i128>();
    check_ir::<PcuU256>();
    check_ir::<PcuI256>();
    check_ir::<PcuU512>();
    check_ir::<PcuI512>();
}

fn native_call<T: PcuCheckedInteger + PartialEq + core::fmt::Debug>(one: T, two: T, maximum: T) {
    let backend = fusion_pcu_cpu::PcuCpuHostBackend::scalar();
    let mut add = sum_prepare::<T, 7, _>(&backend).unwrap();
    let mut output = [maximum; 9];
    add(&[one; 7], &[one; 7], &mut output).unwrap();
    assert_eq!(
        output,
        [two, two, two, two, two, two, two, maximum, maximum]
    );
    let previous = output;
    let mut left = [one; 7];
    left[5] = maximum;
    let error = add(&left, &[one; 7], &mut output).unwrap_err();
    let fault = error.fault().unwrap();
    assert_eq!(fault.invocation_id, 5);
    assert_eq!(
        fault.kind,
        pcu_alias::PcuExecutionFaultKind::ArithmeticOverflow
    );
    assert!(!fault.recovered);
    assert_eq!(output, previous);
    add(&[one; 7], &[one; 7], &mut output).unwrap();
    let mut multiply = scale_prepare::<T, 7, _>(&backend).unwrap();
    multiply(&[one; 7], &two, &mut output).unwrap();
    assert_eq!(output, previous);
    let mut subtract = difference_prepare::<T, 7, _>(&backend).unwrap();
    subtract(&[two; 7], &[one; 7], &mut output).unwrap();
    assert_eq!(
        output,
        [one, one, one, one, one, one, one, maximum, maximum]
    );
    // Matrix specialization preserves the same scalar width and row-major fault
    // index. A failing last cell must not publish earlier successful cells.
    let mut matrix = [[maximum; 3]; 2];
    let mut matrix_add = matrix_sum_prepare::<T, 2, 3, _>(&backend).unwrap();
    matrix_add(&[[one; 3]; 2], &[[one; 3]; 2], &mut matrix).unwrap();
    assert_eq!(matrix, [[two; 3]; 2]);
    let mut matrix_left = [[one; 3]; 2];
    matrix_left[1][2] = maximum;
    let error = matrix_add(&matrix_left, &[[one; 3]; 2], &mut matrix).unwrap_err();
    assert_eq!(error.fault().unwrap().invocation_id, 5);
    assert_eq!(matrix, [[two; 3]; 2]);
    matrix_add(&[[one; 3]; 2], &[[one; 3]; 2], &mut matrix).unwrap();
    let error = multiply(&[maximum; 7], &two, &mut output).unwrap_err();
    assert_eq!(
        error.fault().unwrap().kind,
        pcu_alias::PcuExecutionFaultKind::ArithmeticOverflow
    );
    assert_eq!(
        output,
        [one, one, one, one, one, one, one, maximum, maximum]
    );
}

#[test]
fn eight_native_integer_representations_execute_transactionally_and_retry() {
    native_call(1_u8, 2, u8::MAX);
    native_call(1_i8, 2, i8::MAX);
    native_call(1_u16, 2, u16::MAX);
    native_call(1_i16, 2, i16::MAX);
    native_call(1_u32, 2, u32::MAX);
    native_call(1_i32, 2, i32::MAX);
    native_call(1_u64, 2, u64::MAX);
    native_call(1_i64, 2, i64::MAX);
}

#[test]
fn six_wide_integer_representations_execute_without_narrowing() {
    native_call(1_u128, 2, u128::MAX);
    native_call(1_i128, 2, i128::MAX);
    native_call(
        PcuU256::from_limbs_le([1, 0, 0, 0]),
        PcuU256::from_limbs_le([2, 0, 0, 0]),
        PcuU256::MAX,
    );
    native_call(
        PcuI256::from_limbs_le([1, 0, 0, 0]),
        PcuI256::from_limbs_le([2, 0, 0, 0]),
        PcuI256::MAX,
    );
    native_call(
        PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]),
        PcuU512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]),
        PcuU512::MAX,
    );
    native_call(
        PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]),
        PcuI512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]),
        PcuI512::MAX,
    );
}

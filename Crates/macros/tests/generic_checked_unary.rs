//! Scalar source intrinsics preserve the same six-format typed operation contract.
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    validate_checked_float_map_kernel,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuDispatchDataOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchOp,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
};

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn flip<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

#[pcu(invocations = 3, crate_path = ::pcu_alias, flag(clamp_range), flag(reject_subnormal_result))]
fn activate<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn qualified<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu_alias::pcu::relu(input[id]);
}

fn check_ir<T: PcuCheckedFloat>() {
    let bindings = flip_bindings::<T>();
    let builder = flip_ir::<T, 7>(&bindings).unwrap();
    let kernel = builder.ir();
    validate_checked_float_map_kernel(
        &kernel,
        PcuValueType::Scalar(T::TYPE),
        PcuValueTypeCaps::for_scalar(T::TYPE),
    )
    .unwrap();
    assert!(kernel.ops.iter().any(|op| matches!(op,
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            op: PcuDispatchFloatUnaryOp::Neg, value_type, range_policy: PcuRangePolicy::Reject, ..
        }) if *value_type == PcuValueType::Scalar(T::TYPE))));
    let bindings = activate_bindings::<T>();
    let builder = activate_ir::<T, 7>(&bindings).unwrap();
    builder.with_ir(|kernel| {
        validate_checked_float_map_kernel(kernel, PcuValueType::Scalar(T::TYPE), PcuValueTypeCaps::for_scalar(T::TYPE)).unwrap();
        let PcuDispatchOp::GridStrideLoop { body, .. } = kernel.ops[0] else { panic!("expected canonical grid"); };
        assert!(body.iter().any(|op| matches!(op,
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                op: PcuDispatchFloatUnaryOp::Relu, value_type, range_policy: PcuRangePolicy::Clamp, ..
            }) if *value_type == PcuValueType::Scalar(T::TYPE))));
    });
    let bindings = qualified_bindings::<T>();
    let builder = qualified_ir::<T, 7>(&bindings).unwrap();
    validate_checked_float_map_kernel(
        &builder.ir(),
        PcuValueType::Scalar(T::TYPE),
        PcuValueTypeCaps::for_scalar(T::TYPE),
    )
    .unwrap();
}

#[test]
fn all_six_formats_lower_exact_neg_and_reserved_relu() {
    check_ir::<PcuF16Bits>();
    check_ir::<PcuBf16Bits>();
    check_ir::<PcuF8E4M3FnBits>();
    check_ir::<PcuF8E5M2Bits>();
    check_ir::<f32>();
    check_ir::<f64>();
}

#[test]
fn existing_native_neg_keeps_its_signed_zero_and_transactional_behavior() {
    let backend = fusion_pcu_cpu::PcuCpuHostBackend::scalar();
    let mut call = flip_prepare::<f64, 3, _>(&backend).unwrap();
    let mut output = [7.0; 4];
    call(&[0.0, -0.0, f64::from_bits(1)], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [
            (-0.0_f64).to_bits(),
            0,
            (1_u64 << 63) | 1,
            7.0_f64.to_bits()
        ]
    );
    let previous = output.map(f64::to_bits);
    assert!(call(&[1.0, f64::NAN, 2.0], &mut output).is_err());
    assert_eq!(output.map(f64::to_bits), previous);
}

//! Specialization preserves exact scalar types and numerical policies, without
//! pretending that transporting a format establishes backend arithmetic support.
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    validate_typed_dispatch_value_flow,
    PcuCheckedFloat,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchOp,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuValueType,
};

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn product<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * right[id];
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn scale<T: PcuCheckedFloat, const N: usize>(input: &[T], factor: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] * *factor;
}

#[pcu(invocations = 3, crate_path = ::pcu_alias, flag(reject_subnormal_result), flag(clamp_range))]
fn quotient<T, const N: usize>(left: &[T], right: &[T], output: &mut [T])
where
    T: PcuCheckedFloat,
{
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn nested<T: PcuCheckedFloat, const R: usize, const C: usize>(
    left: &[[T; C]; R],
    right: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = context.global_invocation_id;
    output[id / C][id % C] = -(left[id / C][id % C] + right[id / C][id % C]);
}

fn check_specialization<T: PcuCheckedFloat>() {
    let bindings = product_bindings::<T>();
    let builder = product_ir::<T, 7>(&bindings).unwrap();
    let kernel = builder.ir();
    validate_typed_dispatch_value_flow(&kernel).unwrap();
    assert!(kernel.ops.iter().any(|op| matches!(op,
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type, op: PcuDispatchFloatBinaryOp::Mul,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: PcuRangePolicy::Reject, ..
        }) if *value_type == PcuValueType::Scalar(T::TYPE))));
    let bindings = quotient_bindings::<T>();
    let builder = quotient_ir::<T, 7>(&bindings).unwrap();
    builder.with_ir(|kernel| {
        validate_typed_dispatch_value_flow(kernel).unwrap();
        let PcuDispatchOp::GridStrideLoop { body, extent: 7 } = kernel.ops[0] else {
            panic!("expected specialized grid region")
        };
        assert!(body.iter().any(|op| matches!(op,
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                value_type, op: PcuDispatchFloatBinaryOp::Div,
                underflow_policy: PcuFloatUnderflowPolicy::RejectSubnormalResult,
                range_policy: PcuRangePolicy::Clamp, ..
            }) if *value_type == PcuValueType::Scalar(T::TYPE))));
    });
    let bindings = nested_bindings::<T>();
    let builder = nested_ir::<T, 2, 3>(&bindings).unwrap();
    validate_typed_dispatch_value_flow(&builder.ir()).unwrap();
    let bindings = scale_bindings::<T>();
    let builder = scale_ir::<T, 7>(&bindings).unwrap();
    let kernel = builder.ir();
    validate_typed_dispatch_value_flow(&kernel).unwrap();
    assert_eq!(
        kernel.minimum_binding_elements_for(pcu_alias::PcuBindingRef::new(0, 1), 7),
        1
    );
}

#[test]
fn binary16_ir() {
    check_specialization::<PcuF16Bits>();
}
#[test]
fn bf16_ir() {
    check_specialization::<PcuBf16Bits>();
}
#[test]
fn binary32_ir() {
    check_specialization::<f32>();
}
#[test]
fn binary64_ir() {
    check_specialization::<f64>();
}
#[test]
fn e4m3fn_ir() {
    check_specialization::<PcuF8E4M3FnBits>();
}
#[test]
fn e5m2_ir() {
    check_specialization::<PcuF8E5M2Bits>();
}

#[test]
fn primitive_prepared_calls_are_typed_transactional_and_reusable() {
    use fusion_pcu_cpu::PcuCpuHostBackend;
    let backend = PcuCpuHostBackend::scalar();
    let mut call = product_prepare::<f32, 3, _>(&backend).unwrap();
    let mut output = [99.0_f32; 5];
    call(&[2.0, -3.0, -0.0], &[4.0, 5.0, 1.0], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [8.0, -15.0, -0.0, 99.0, 99.0].map(f32::to_bits)
    );
    let previous = output.map(f32::to_bits);
    let error = call(&[2.0, f32::MAX, 1.0], &[4.0, 2.0, 1.0], &mut output).unwrap_err();
    assert_eq!(error.fault().unwrap().invocation_id, 1);
    assert_eq!(output.map(f32::to_bits), previous);
    call(&[1.0; 3], &[2.0; 3], &mut output).unwrap();
    let mut double_call = product_prepare::<f64, 3, _>(&backend).unwrap();
    let mut double_output = [0.0_f64; 3];
    double_call(
        &[1.0, f64::from_bits(1), -0.0],
        &[2.0; 3],
        &mut double_output,
    )
    .unwrap();
    assert_eq!(
        double_output.map(f64::to_bits),
        [2.0_f64.to_bits(), 2, (-0.0_f64).to_bits()]
    );
    let mut broadcast = scale_prepare::<f64, 3, _>(&backend).unwrap();
    broadcast(&[1.0, -2.0, -0.0], &3.0, &mut double_output).unwrap();
    assert_eq!(
        double_output.map(f64::to_bits),
        [3.0_f64, -6.0, -0.0].map(f64::to_bits)
    );
}

#[test]
fn low_precision_prepared_calls_publish_only_complete_checked_results() {
    use fusion_pcu_cpu::PcuCpuHostBackend;
    fn check<T: PcuCheckedFloat + PartialEq + core::fmt::Debug>(one: T, two: T, maximum: T) {
        let backend = PcuCpuHostBackend::scalar();
        let mut call = product_prepare::<T, 3, _>(&backend).unwrap();
        let mut output = [maximum; 5];
        call(&[one; 3], &[two; 3], &mut output).unwrap();
        assert_eq!(output, [two, two, two, maximum, maximum]);
        let previous = output;
        let fault = call(&[one, maximum, one], &[two; 3], &mut output).unwrap_err();
        assert_eq!(fault.fault().unwrap().invocation_id, 1);
        assert_eq!(output, previous);
        call(&[one; 3], &[one; 3], &mut output).unwrap();
        assert_eq!(output, [one, one, one, maximum, maximum]);
        let mut broadcast = scale_prepare::<T, 3, _>(&backend).unwrap();
        broadcast(&[one; 3], &two, &mut output).unwrap();
        assert_eq!(output, [two, two, two, maximum, maximum]);
    }
    let backend = PcuCpuHostBackend::scalar();
    check(
        PcuF16Bits::from_bits(0x3c00),
        PcuF16Bits::from_bits(0x4000),
        PcuF16Bits::from_bits(0x7bff),
    );
    check(
        PcuBf16Bits::from_bits(0x3f80),
        PcuBf16Bits::from_bits(0x4000),
        PcuBf16Bits::from_bits(0x7f7f),
    );
    check(
        PcuF8E4M3FnBits::from_bits(0x38),
        PcuF8E4M3FnBits::from_bits(0x40),
        PcuF8E4M3FnBits::from_bits(0x7e),
    );
    check(
        PcuF8E5M2Bits::from_bits(0x3c),
        PcuF8E5M2Bits::from_bits(0x40),
        PcuF8E5M2Bits::from_bits(0x7b),
    );
    // A binary implementation does not imply unary recovery or composed tensor support.
    assert!(nested_prepare::<PcuF16Bits, 2, 3, _>(&backend).is_err());
}

#[test]
fn generic_grid_clamp_publishes_completed_output_and_preserves_fatal_rollback() {
    use fusion_pcu_cpu::PcuCpuHostBackend;
    let backend = PcuCpuHostBackend::scalar();
    let one = PcuF16Bits::from_bits(0x3c00);
    let tiny = PcuF16Bits::from_bits(1);
    let zero = PcuF16Bits::from_bits(0);
    let sentinel = PcuF16Bits::from_bits(0x4000);
    let mut call = quotient_prepare::<PcuF16Bits, 7, _>(&backend).unwrap();
    let mut output = [sentinel; 9];
    let fault = call(&[tiny; 7], &[one; 7], &mut output).unwrap_err();
    let fault = fault.fault().unwrap();
    assert!(fault.recovered);
    assert_eq!(fault.invocation_id, 0);
    assert_eq!(
        fault.kind,
        pcu_alias::PcuExecutionFaultKind::ArithmeticUnderflow
    );
    assert_eq!(
        output,
        [tiny, tiny, tiny, tiny, tiny, tiny, tiny, sentinel, sentinel]
    );

    // An earlier recoverable lane must not conceal a later fatal lane.
    let previous = output;
    let mut divisors = [one; 7];
    divisors[4] = zero;
    let fault = call(&[tiny; 7], &divisors, &mut output).unwrap_err();
    let fault = fault.fault().unwrap();
    assert!(!fault.recovered);
    assert_eq!(fault.invocation_id, 4);
    assert_eq!(fault.kind, pcu_alias::PcuExecutionFaultKind::DivideByZero);
    assert_eq!(output, previous);

    call(&[one; 7], &[one; 7], &mut output).unwrap();
    assert_eq!(
        output,
        [one, one, one, one, one, one, one, sentinel, sentinel]
    );
}

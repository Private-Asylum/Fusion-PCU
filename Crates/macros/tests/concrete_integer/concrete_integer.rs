use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    PcuDispatchDataOp,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchOp,
    PcuExecutionFaultKind,
    PcuCheckedInteger,
    PcuRangePolicy,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
    assess_checked_integer_binary_operands,
    validate_typed_dispatch_value_flow,
};

#[pcu(invocations = 7, crate_path = ::pcu_alias)]
fn repeated(output: &mut [i128], input: &[i128]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}

#[pcu(invocations = 7, crate_path = ::pcu_alias)]
fn independent(input: &[i128], output: &mut [i128]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0] - input[id];
}

#[pcu(invocations = 3, flag(clamp_range), crate_path = ::pcu_alias)]
fn clamped(input: &[u128], factor: &u128, output: &mut [u128]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = input[id] * *factor;
        id += stride;
    }
}

#[pcu(invocations = 7, crate_path = ::pcu_alias)]
fn identity(input: &[u128], output: &mut [u128]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[pcu(invocations = 7, crate_path = ::pcu_alias)]
fn generic_repeated<T: PcuCheckedInteger>(output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}

#[pcu(invocations = 3, flag(clamp_range), crate_path = ::pcu_alias)]
fn generic_clamped<T: PcuCheckedInteger>(input: &[T], factor: &T, output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = input[id] * *factor;
        id += stride;
    }
}

#[test]
fn concrete_and_generic_spellings_emit_identical_ir() {
    let concrete_bindings = repeated_bindings();
    let concrete = repeated_ir(&concrete_bindings).unwrap();
    let generic_bindings = generic_repeated_bindings::<i128>();
    let generic = generic_repeated_ir::<i128>(&generic_bindings).unwrap();
    assert_eq!(concrete.ir(), generic.ir());
    let concrete_bindings = clamped_bindings();
    let concrete = clamped_ir(&concrete_bindings).unwrap();
    let generic_bindings = generic_clamped_bindings::<u128>();
    let generic = generic_clamped_ir::<u128>(&generic_bindings).unwrap();
    concrete.with_ir(|kernel| generic.with_ir(|other| assert_eq!(kernel, other)));
}

#[test]
fn concrete_checked_source_preserves_exact_type_and_range_ir() {
    let bindings = repeated_bindings();
    let builder = repeated_ir(&bindings).unwrap();
    let kernel = builder.ir();
    validate_typed_dispatch_value_flow(&kernel).unwrap();
    assess_checked_integer_binary_operands(
        &kernel,
        PcuValueType::Scalar(PcuScalarType::I128),
        PcuDispatchIntegerBinaryOp::Add,
        PcuValueTypeCaps::INT128,
    )
    .unwrap();
    assert_eq!(
        kernel.numerical_requirements.range_policy,
        PcuRangePolicy::Reject
    );
    let bindings = clamped_bindings();
    let builder = clamped_ir(&bindings).unwrap();
    builder.with_ir(|kernel| {
        validate_typed_dispatch_value_flow(kernel).unwrap();
        assess_checked_integer_binary_operands(
            kernel,
            PcuValueType::Scalar(PcuScalarType::U128),
            PcuDispatchIntegerBinaryOp::Mul,
            PcuValueTypeCaps::UINT128,
        )
        .unwrap();
        assert_eq!(
            kernel.numerical_requirements.range_policy,
            PcuRangePolicy::Clamp
        );
        let Some(PcuDispatchOp::GridStrideLoop { body, .. }) = kernel.ops.first() else {
            panic!("grid source must retain its execution hierarchy");
        };
        assert!(body.iter().any(|op| matches!(
            op,
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                range_policy: PcuRangePolicy::Clamp,
                ..
            })
        )));
    });
}

#[test]
fn signed_full_width_inputs_keep_faults_private_and_retry() {
    let backend = fusion_pcu_cpu::PcuCpuHostBackend::scalar();
    let mut add = repeated_prepare(&backend).unwrap();
    let high = 1_i128 << 100;
    let mut output = [99_i128; 9];
    add(&mut output, &[high; 7]).unwrap();
    assert_eq!(
        output,
        [
            high * 2,
            high * 2,
            high * 2,
            high * 2,
            high * 2,
            high * 2,
            high * 2,
            99,
            99
        ]
    );
    let before = output;
    for (endpoint, kind) in [
        (i128::MAX, PcuExecutionFaultKind::ArithmeticOverflow),
        (i128::MIN, PcuExecutionFaultKind::ArithmeticUnderflow),
    ] {
        let mut input = [high; 7];
        input[2] = endpoint;
        input[6] = endpoint;
        let error = add(&mut output, &input).unwrap_err();
        let fault = error.fault().unwrap();
        assert_eq!(fault.kind, kind);
        assert_eq!(fault.invocation_id, 2);
        assert!(!fault.recovered);
        assert_eq!(output, before);
        add(&mut output, &[high; 7]).unwrap();
        assert_eq!(output, before);
    }
    let mut subtract = independent_prepare(&backend).unwrap();
    let input = [
        high,
        high + 1,
        high + 2,
        high + 3,
        high + 4,
        high + 5,
        high + 6,
    ];
    subtract(&input, &mut output).unwrap();
    assert_eq!(output, [0, -1, -2, -3, -4, -5, -6, 99, 99]);
}

#[test]
fn unsigned_grid_borrowed_factor_clamps_observably_and_identity_keeps_all_bits() {
    let backend = fusion_pcu_cpu::PcuCpuHostBackend::scalar();
    let mut multiply = clamped_prepare(&backend).unwrap();
    let high = 1_u128 << 100;
    let mut input = [high; 7];
    input[2] = u128::MAX;
    input[6] = u128::MAX;
    let mut output = [99_u128; 9];
    let error = multiply(&input, &2, &mut output).unwrap_err();
    let fault = error.fault().unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(fault.invocation_id, 2);
    assert!(fault.recovered);
    assert_eq!(
        output,
        [
            high * 2,
            high * 2,
            u128::MAX,
            high * 2,
            high * 2,
            high * 2,
            u128::MAX,
            99,
            99
        ]
    );
    multiply(&[high; 7], &2, &mut output).unwrap();
    assert_eq!(&output[..7], &[high * 2; 7]);
    let mut copy = identity_prepare(&backend).unwrap();
    input[0] = 0;
    input[3] = 1_u128 << 127;
    copy(&input, &mut output).unwrap();
    assert_eq!(&output[..7], &input);
    assert_eq!(&output[7..], &[99; 2]);
}

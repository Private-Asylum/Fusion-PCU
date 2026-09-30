//! Actual annotated source lowers to checked scalar contracts.
use core::num::NonZeroU32;
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCheckedFloatReference,
    PcuCpuTypedBinding,
    PcuCpuTypedSlice,
};
#[rustfmt::skip]
use pcu_alias::{
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuInvocationShape,
};
#[pcu(invocations=4, crate_path=::pcu_alias)]
fn negate(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = -input[invocation];
}
#[pcu(crate_path=::pcu_alias)]
fn flip(value: f64) -> f64 {
    -value
}
#[pcu(invocations=4, crate_path=::pcu_alias)]
fn helper_negate(input: &[f64], output: &mut [f64]) {
    let invocation = context.global_invocation_id;
    output[invocation] = flip(input[invocation]);
}
#[test]
fn annotated_negation_executes_signed_zero_and_subnormals() {
    let descriptors = negate_bindings();
    let builder = negate_ir(&descriptors).unwrap();
    let kernel = builder.ir();
    assert!(kernel.ops.iter().any(|op| matches!(
        op,
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            op: PcuDispatchFloatUnaryOp::Neg,
            ..
        })
    )));
    let input = [0.0_f32, -0.0, f32::from_bits(1), f32::MAX];
    let mut output = [9.0_f32; 4];
    let mut bindings = [
        PcuCpuTypedBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuCpuTypedSlice::ReadF32(&input),
        },
        PcuCpuTypedBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuCpuTypedSlice::ReadWriteF32(&mut output),
        },
    ];
    PcuCheckedFloatReference
        .run_host_direct(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(4).unwrap()),
            },
            &mut bindings,
        )
        .unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        input.map(|value| value.to_bits() ^ 0x8000_0000)
    );
    let descriptors = helper_negate_bindings();
    let builder = helper_negate_ir(&descriptors).unwrap();
    let kernel = builder.ir();
    let input = [0.0_f64, -0.0, f64::from_bits(1), f64::MAX];
    let mut output = [9.0_f64; 4];
    let mut bindings = [
        PcuCpuTypedBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuCpuTypedSlice::ReadF64(&input),
        },
        PcuCpuTypedBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuCpuTypedSlice::ReadWriteF64(&mut output),
        },
    ];
    PcuCheckedFloatReference
        .run_host_direct(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(4).unwrap()),
            },
            &mut bindings,
        )
        .unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        input.map(|value| value.to_bits() ^ (1 << 63))
    );
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u8_add(input: &[u8], rhs: &[u8], output: &mut [u8]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u8_sub(input: &[u8], rhs: &[u8], output: &mut [u8]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] - rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u8_mul(input: &[u8], rhs: &[u8], output: &mut [u8]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u16_add(input: &[u16], rhs: &[u16], output: &mut [u16]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u16_sub(input: &[u16], rhs: &[u16], output: &mut [u16]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] - rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u16_mul(input: &[u16], rhs: &[u16], output: &mut [u16]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u32_add(input: &[u32], rhs: &[u32], output: &mut [u32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u32_sub(input: &[u32], rhs: &[u32], output: &mut [u32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] - rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u32_mul(input: &[u32], rhs: &[u32], output: &mut [u32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u64_add(input: &[u64], rhs: &[u64], output: &mut [u64]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u64_sub(input: &[u64], rhs: &[u64], output: &mut [u64]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] - rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn u64_mul(input: &[u64], rhs: &[u64], output: &mut [u64]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i8_add(input: &[i8], rhs: &[i8], output: &mut [i8]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i8_sub(input: &[i8], rhs: &[i8], output: &mut [i8]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] - rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i8_mul(input: &[i8], rhs: &[i8], output: &mut [i8]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i16_add(input: &[i16], rhs: &[i16], output: &mut [i16]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i16_sub(input: &[i16], rhs: &[i16], output: &mut [i16]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] - rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i16_mul(input: &[i16], rhs: &[i16], output: &mut [i16]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i32_add(input: &[i32], rhs: &[i32], output: &mut [i32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i32_sub(input: &[i32], rhs: &[i32], output: &mut [i32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] - rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i32_mul(input: &[i32], rhs: &[i32], output: &mut [i32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i64_add(input: &[i64], rhs: &[i64], output: &mut [i64]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i64_sub(input: &[i64], rhs: &[i64], output: &mut [i64]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] - rhs[invocation];
}

#[pcu(invocations=2, crate_path=::pcu_alias)]
fn i64_mul(input: &[i64], rhs: &[i64], output: &mut [i64]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * rhs[invocation];
}

fn assert_checked_integer(
    kernel: &pcu_alias::PcuDispatchKernelIr<'_>,
    value_type: pcu_alias::PcuValueType,
    operation: PcuDispatchIntegerBinaryOp,
) {
    pcu_alias::validate_typed_dispatch_value_flow(kernel).unwrap();
    assert!(kernel.ops.iter().any(|op|matches!(op,PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary{value_type:actual,op:actual_op,..}) if *actual==value_type && *actual_op==operation)));
    assert!(
        !kernel
            .ops
            .iter()
            .any(|op| matches!(op, PcuDispatchOp::Data(PcuDispatchDataOp::Alu { .. })))
    );
}

#[test]
fn u8_ordinary_source_uses_checked_ir() {
    let descriptors = u8_add_bindings();
    let builder = u8_add_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u8(),
        PcuDispatchIntegerBinaryOp::Add,
    );
    let descriptors = u8_sub_bindings();
    let builder = u8_sub_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u8(),
        PcuDispatchIntegerBinaryOp::Sub,
    );
    let descriptors = u8_mul_bindings();
    let builder = u8_mul_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u8(),
        PcuDispatchIntegerBinaryOp::Mul,
    );
}

#[test]
fn u16_ordinary_source_uses_checked_ir() {
    let descriptors = u16_add_bindings();
    let builder = u16_add_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u16(),
        PcuDispatchIntegerBinaryOp::Add,
    );
    let descriptors = u16_sub_bindings();
    let builder = u16_sub_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u16(),
        PcuDispatchIntegerBinaryOp::Sub,
    );
    let descriptors = u16_mul_bindings();
    let builder = u16_mul_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u16(),
        PcuDispatchIntegerBinaryOp::Mul,
    );
}

#[test]
fn u32_ordinary_source_uses_checked_ir() {
    let descriptors = u32_add_bindings();
    let builder = u32_add_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u32(),
        PcuDispatchIntegerBinaryOp::Add,
    );
    let descriptors = u32_sub_bindings();
    let builder = u32_sub_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u32(),
        PcuDispatchIntegerBinaryOp::Sub,
    );
    let descriptors = u32_mul_bindings();
    let builder = u32_mul_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u32(),
        PcuDispatchIntegerBinaryOp::Mul,
    );
}

#[test]
fn u64_ordinary_source_uses_checked_ir() {
    let descriptors = u64_add_bindings();
    let builder = u64_add_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u64(),
        PcuDispatchIntegerBinaryOp::Add,
    );
    let descriptors = u64_sub_bindings();
    let builder = u64_sub_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u64(),
        PcuDispatchIntegerBinaryOp::Sub,
    );
    let descriptors = u64_mul_bindings();
    let builder = u64_mul_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::u64(),
        PcuDispatchIntegerBinaryOp::Mul,
    );
}

#[test]
fn i8_ordinary_source_uses_checked_ir() {
    let descriptors = i8_add_bindings();
    let builder = i8_add_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i8(),
        PcuDispatchIntegerBinaryOp::Add,
    );
    let descriptors = i8_sub_bindings();
    let builder = i8_sub_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i8(),
        PcuDispatchIntegerBinaryOp::Sub,
    );
    let descriptors = i8_mul_bindings();
    let builder = i8_mul_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i8(),
        PcuDispatchIntegerBinaryOp::Mul,
    );
}

#[test]
fn i16_ordinary_source_uses_checked_ir() {
    let descriptors = i16_add_bindings();
    let builder = i16_add_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i16(),
        PcuDispatchIntegerBinaryOp::Add,
    );
    let descriptors = i16_sub_bindings();
    let builder = i16_sub_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i16(),
        PcuDispatchIntegerBinaryOp::Sub,
    );
    let descriptors = i16_mul_bindings();
    let builder = i16_mul_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i16(),
        PcuDispatchIntegerBinaryOp::Mul,
    );
}

#[test]
fn i32_ordinary_source_uses_checked_ir() {
    let descriptors = i32_add_bindings();
    let builder = i32_add_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i32(),
        PcuDispatchIntegerBinaryOp::Add,
    );
    let descriptors = i32_sub_bindings();
    let builder = i32_sub_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i32(),
        PcuDispatchIntegerBinaryOp::Sub,
    );
    let descriptors = i32_mul_bindings();
    let builder = i32_mul_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i32(),
        PcuDispatchIntegerBinaryOp::Mul,
    );
}

#[test]
fn i64_ordinary_source_uses_checked_ir() {
    let descriptors = i64_add_bindings();
    let builder = i64_add_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i64(),
        PcuDispatchIntegerBinaryOp::Add,
    );
    let descriptors = i64_sub_bindings();
    let builder = i64_sub_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i64(),
        PcuDispatchIntegerBinaryOp::Sub,
    );
    let descriptors = i64_mul_bindings();
    let builder = i64_mul_ir(&descriptors).unwrap();
    assert_checked_integer(
        &builder.ir(),
        pcu_alias::PcuValueType::i64(),
        PcuDispatchIntegerBinaryOp::Mul,
    );
}

#[pcu(invocations=2,crate_path=::pcu_alias)]
fn integer_literals(input: &[i8], output: &mut [i8]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + -128;
}
#[test]
fn signed_minimum_literal_is_inferred_without_a_widening_cast() {
    let descriptors = integer_literals_bindings();
    let builder = integer_literals_ir(&descriptors).unwrap();
    let kernel = builder.ir();
    pcu_alias::validate_typed_dispatch_value_flow(&kernel).unwrap();
    assert!(kernel.ops.iter().any(|op| matches!(
        op,
        PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
            value: pcu_alias::PcuParameterValue::I8(-128),
            ..
        })
    )));
}

#[pcu(invocations=4,crate_path=::pcu_alias,flag(reject_subnormal_result),flag(clamp_range))]
fn clamped_negate(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = -input[invocation];
}
#[test]
fn annotated_negation_reports_recovery_and_preserves_complete_output() {
    let descriptors = clamped_negate_bindings();
    let builder = clamped_negate_ir(&descriptors).unwrap();
    let kernel = builder.ir();
    let input = [1.0_f32, f32::from_bits(1), -0.0, -2.0];
    let mut output = [9.0_f32; 4];
    let mut bindings = [
        PcuCpuTypedBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuCpuTypedSlice::ReadF32(&input),
        },
        PcuCpuTypedBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuCpuTypedSlice::ReadWriteF32(&mut output),
        },
    ];
    let error = PcuCheckedFloatReference
        .run_host_direct(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(4).unwrap()),
            },
            &mut bindings,
        )
        .unwrap_err();
    let fusion_pcu_cpu::PcuCheckedFloatReferenceError::Fault(fault) = error else {
        panic!("observable range fault required")
    };
    assert_eq!(
        fault.kind,
        pcu_alias::PcuExecutionFaultKind::ArithmeticUnderflow
    );
    assert!(fault.recovered);
    assert_eq!(fault.invocation_id, 1);
    assert_eq!(
        output.map(f32::to_bits),
        input.map(|value| value.to_bits() ^ 0x8000_0000)
    );
}

//! Hardware acceptance for checked floating-point unary operations.

use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchEntryPoint,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuInvocationShape,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
};
use std::num::NonZeroU32;

fn prepare_unary(
    operation: PcuDispatchFloatUnaryOp,
    backend: &RocmOwnedDispatchBackend,
    value_type: PcuValueType,
    policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
    grid: bool,
) -> RocmPreparedDispatch {
    let bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
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
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type,
            op: operation,
            underflow_policy: policy,
            range_policy,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index,
            value: PcuDispatchValueId(2),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let loop_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 4,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    backend
        .prepare_dispatch(PcuDispatchSubmission {
            kernel: &PcuDispatchKernelIr {
                numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                id: fusion_pcu::PcuKernelId(0xf17e),
                entry: PcuDispatchEntryPoint {
                    name: "rocm_checked_unary",
                    logical_shape: [if grid { 1 } else { 4 }, 1, 1],
                },
                bindings: &bindings,
                ports: &[],
                parameters: &[],
                ops: if grid { &loop_ops } else { &direct },
                type_caps: PcuValueTypeCaps::for_scalar(value_type.scalar_type())
                    .union(PcuValueTypeCaps::SCALAR_VALUES),
                feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
                    .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES)
                    .union(if range_policy == PcuRangePolicy::Clamp {
                        PcuDispatchFeatureCaps::RANGE_CLAMP
                    } else {
                        PcuDispatchFeatureCaps::empty()
                    }),
            },
            shape: PcuInvocationShape::invocations(
                NonZeroU32::new(if grid { 1 } else { 4 }).expect("nonempty shape"),
            ),
        })
        .expect("prepare checked unary")
}

fn run_unary(
    backend: &RocmOwnedDispatchBackend,
    prepared: &RocmPreparedDispatch,
    value_type: PcuValueType,
    input: &[u8],
    output_sentinel: &[u8],
    fault_word: &mut DeviceBuffer,
) -> (PcuCompletionOutcome, DeviceBuffer) {
    let mut input_buffer = backend.allocate(input.len()).unwrap();
    input_buffer.copy_from(input).unwrap();
    let mut output = backend.allocate(output_sentinel.len()).unwrap();
    output.copy_from(output_sentinel).unwrap();
    let bindings = [
        backend
            .binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(value_type),
                input_buffer,
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(value_type),
                output.clone(),
            )
            .unwrap(),
    ];
    let mut completion = prepared
        .submit_with_fault_word(&bindings, fault_word)
        .expect("submit checked unary");
    (completion.wait().expect("wait for checked unary"), output)
}

fn encode_f32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|bits| bits.to_ne_bytes()).collect()
}

fn encode_f64(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|bits| bits.to_ne_bytes()).collect()
}

fn assert_sequential_unary_lifecycle(
    backend: &RocmOwnedDispatchBackend,
    prepared: &RocmPreparedDispatch,
    value_type: PcuValueType,
    finite_input: &[u8],
    finite_expected: &[u8],
    subnormal_input: &[u8],
    recovered_expected: &[u8],
) {
    let mut input = backend.allocate(finite_input.len()).unwrap();
    let mut output = backend.allocate(finite_expected.len()).unwrap();
    let bindings = [
        backend
            .binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(value_type),
                input.clone(),
            )
            .unwrap(),
        backend
            .binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(value_type),
                output.clone(),
            )
            .unwrap(),
    ];
    let mut sequential = prepared.sequential_checked().unwrap();
    let sentinel = vec![0x5a; finite_expected.len()];
    let mut actual = vec![0; finite_expected.len()];
    // A certain binding rejection must leave the owner available for a valid retry.
    assert!(matches!(
        sequential.submit_and_wait(&[]),
        Err(RocmOwnedDispatchError::Binding(_))
    ));
    input.copy_from(finite_input).unwrap();
    for _ in 0..2 {
        output.copy_from(&sentinel).unwrap();
        assert_eq!(
            sequential.submit_and_wait(&bindings).unwrap(),
            PcuCompletionOutcome::Succeeded
        );
        output.copy_to(&mut actual).unwrap();
        assert_eq!(actual, finite_expected);
    }
    input.copy_from(subnormal_input).unwrap();
    output.copy_from(&sentinel).unwrap();
    assert_eq!(
        sequential.submit_and_wait(&bindings).unwrap(),
        PcuCompletionOutcome::Fault(fusion_pcu::PcuExecutionFault {
            kind: fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow,
            invocation_id: 0,
            recovered: true,
        })
    );
    output.copy_to(&mut actual).unwrap();
    assert_eq!(actual, recovered_expected);
    input.copy_from(finite_input).unwrap();
    output.copy_from(&sentinel).unwrap();
    assert_eq!(
        sequential.submit_and_wait(&bindings).unwrap(),
        PcuCompletionOutcome::Succeeded
    );
    output.copy_to(&mut actual).unwrap();
    assert_eq!(actual, finite_expected);
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keep typed payloads, fault precedence, and retry in one hardware lifecycle.
fn assert_relu_width(
    backend: &RocmOwnedDispatchBackend,
    value_type: PcuValueType,
    scalar_bytes: usize,
    finite_input: &[u8],
    finite_expected: &[u8],
    subnormal_input: &[u8],
    nan_input: &[u8],
    fatal_input: &[u8],
    recovered_expected: &[u8],
    grid: bool,
) {
    let sentinel = vec![0x5a; scalar_bytes * 4];
    let mut fault_word = backend.allocate(core::mem::size_of::<u64>()).unwrap();
    let ordinary = prepare_unary(
        PcuDispatchFloatUnaryOp::Relu,
        backend,
        value_type,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
        grid,
    );
    let (outcome, output) = run_unary(
        backend,
        &ordinary,
        value_type,
        finite_input,
        &sentinel,
        &mut fault_word,
    );
    assert_eq!(outcome, PcuCompletionOutcome::Succeeded);
    let mut bytes = vec![0; sentinel.len()];
    output.copy_to(&mut bytes).unwrap();
    assert_eq!(bytes, finite_expected);

    let strict = prepare_unary(
        PcuDispatchFloatUnaryOp::Relu,
        backend,
        value_type,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuRangePolicy::Reject,
        grid,
    );
    let (outcome, _) = run_unary(
        backend,
        &strict,
        value_type,
        subnormal_input,
        &sentinel,
        &mut fault_word,
    );
    assert_eq!(
        outcome,
        PcuCompletionOutcome::Fault(fusion_pcu::PcuExecutionFault {
            kind: fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow,
            invocation_id: 0,
            recovered: false,
        })
    );

    let clamp = prepare_unary(
        PcuDispatchFloatUnaryOp::Relu,
        backend,
        value_type,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuRangePolicy::Clamp,
        grid,
    );
    let (outcome, output) = run_unary(
        backend,
        &clamp,
        value_type,
        subnormal_input,
        &sentinel,
        &mut fault_word,
    );
    assert_eq!(
        outcome,
        PcuCompletionOutcome::Fault(fusion_pcu::PcuExecutionFault {
            kind: fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow,
            invocation_id: 0,
            recovered: true,
        })
    );
    output.copy_to(&mut bytes).unwrap();
    assert_eq!(bytes, recovered_expected);

    assert_sequential_unary_lifecycle(
        backend,
        &clamp,
        value_type,
        finite_input,
        finite_expected,
        subnormal_input,
        recovered_expected,
    );

    for (bad, expected_invocation) in [(nan_input, 0), (fatal_input, 2)] {
        let (outcome, _) = run_unary(backend, &clamp, value_type, bad, &sentinel, &mut fault_word);
        assert_eq!(
            outcome,
            PcuCompletionOutcome::Fault(fusion_pcu::PcuExecutionFault {
                kind: fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand,
                invocation_id: expected_invocation,
                recovered: false,
            })
        );
    }

    // The rejected launch remains reusable and the fresh successful result is fully checked.
    let (outcome, output) = run_unary(
        backend,
        &ordinary,
        value_type,
        finite_input,
        &sentinel,
        &mut fault_word,
    );
    assert_eq!(outcome, PcuCompletionOutcome::Succeeded);
    output.copy_to(&mut bytes).unwrap();
    assert_eq!(bytes, finite_expected);
}

#[test]
#[ignore = "requires a working ROCm device"]
fn checked_relu_direct_and_grid_cover_f32_f64_bits_faults_recovery_and_retry() {
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    for grid in [false, true] {
        assert_relu_width(
            &backend,
            PcuValueType::f32(),
            4,
            &encode_f32(&[0x8000_0000, 0xc040_0000, 0x4020_0000, 0x7f7f_ffff]),
            &encode_f32(&[0, 0, 0x4020_0000, 0x7f7f_ffff]),
            &encode_f32(&[1, 0x8000_0001, 0x3f80_0000, 0xbf80_0000]),
            &encode_f32(&[0x7fc0_0001; 4]),
            &encode_f32(&[1, 0x3f80_0000, 0x7f80_0000, 0x3f80_0000]),
            &encode_f32(&[1, 0, 0x3f80_0000, 0]),
            grid,
        );
        assert_relu_width(
            &backend,
            PcuValueType::f64(),
            8,
            &encode_f64(&[
                0x8000_0000_0000_0000,
                0xc008_0000_0000_0000,
                0x4004_0000_0000_0000,
                0x7fef_ffff_ffff_ffff,
            ]),
            &encode_f64(&[0, 0, 0x4004_0000_0000_0000, 0x7fef_ffff_ffff_ffff]),
            &encode_f64(&[
                1,
                0x8000_0000_0000_0001,
                0x3ff0_0000_0000_0000,
                0xbff0_0000_0000_0000,
            ]),
            &encode_f64(&[0x7ff8_0000_0000_0001; 4]),
            &encode_f64(&[
                1,
                0x3ff0_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0x3ff0_0000_0000_0000,
            ]),
            &encode_f64(&[1, 0, 0x3ff0_0000_0000_0000, 0]),
            grid,
        );
    }
}

#[allow(clippy::too_many_lines)] // Keep direct/grid fault precedence, sequential retry, and bit corpus together.
fn assert_neg_width(backend: &RocmOwnedDispatchBackend, value_type: PcuValueType, grid: bool) {
    let width = usize::from(value_type.scalar_type().bit_width()) / 8;
    let sign = if width == 4 { 1u64 << 31 } else { 1u64 << 63 };
    let normal = if width == 4 {
        0x3f80_0000
    } else {
        0x3ff0_0000_0000_0000
    };
    let maximum = if width == 4 {
        0x7f7f_ffff
    } else {
        0x7fef_ffff_ffff_ffff
    };
    let infinity = if width == 4 {
        0x7f80_0000
    } else {
        0x7ff0_0000_0000_0000
    };
    let encode = |values: &[u64]| {
        if width == 4 {
            encode_f32(
                &values
                    .iter()
                    .map(|&v| u32::try_from(v).unwrap())
                    .collect::<Vec<_>>(),
            )
        } else {
            encode_f64(values)
        }
    };
    let finite = [0, sign, maximum, normal];
    let subnormal = [1, sign | 1, normal, normal | sign];
    let expected = encode(&finite.map(|bits| bits ^ sign));
    let recovered = encode(&subnormal.map(|bits| bits ^ sign));
    let sentinel = vec![0x5a; width * 4];
    let mut status = backend.allocate(size_of::<u64>()).unwrap();
    let ordinary = prepare_unary(
        PcuDispatchFloatUnaryOp::Neg,
        backend,
        value_type,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
        grid,
    );
    let reject = prepare_unary(
        PcuDispatchFloatUnaryOp::Neg,
        backend,
        value_type,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuRangePolicy::Reject,
        grid,
    );
    let clamp = prepare_unary(
        PcuDispatchFloatUnaryOp::Neg,
        backend,
        value_type,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuRangePolicy::Clamp,
        grid,
    );
    let check = |prepared: &RocmPreparedDispatch,
                 input: &[u8],
                 expected_outcome: PcuCompletionOutcome,
                 status: &mut DeviceBuffer| {
        let (outcome, output) = run_unary(backend, prepared, value_type, input, &sentinel, status);
        assert_eq!(outcome, expected_outcome);
        let mut bytes = vec![0; sentinel.len()];
        output.copy_to(&mut bytes).unwrap();
        bytes
    };
    assert_eq!(
        check(
            &ordinary,
            &encode(&finite),
            PcuCompletionOutcome::Succeeded,
            &mut status
        ),
        expected
    );
    assert_eq!(
        check(
            &ordinary,
            &encode(&subnormal),
            PcuCompletionOutcome::Succeeded,
            &mut status
        ),
        recovered
    );
    let fault = |kind, invocation_id, recovered| {
        PcuCompletionOutcome::Fault(fusion_pcu::PcuExecutionFault {
            kind,
            invocation_id,
            recovered,
        })
    };
    check(
        &reject,
        &encode(&subnormal),
        fault(PcuExecutionFaultKind::ArithmeticUnderflow, 0, false),
        &mut status,
    );
    assert_eq!(
        check(
            &clamp,
            &encode(&subnormal),
            fault(PcuExecutionFaultKind::ArithmeticUnderflow, 0, true),
            &mut status
        ),
        recovered
    );
    for bad in [infinity, infinity | sign, infinity | 1, infinity | sign | 1] {
        check(
            &ordinary,
            &encode(&[normal, bad, infinity, 0]),
            fault(PcuExecutionFaultKind::InvalidFloatingOperand, 1, false),
            &mut status,
        );
        // A fatal lane supersedes earlier recovered underflow and keeps its own attribution.
        check(
            &clamp,
            &encode(&[1, normal, bad, 0]),
            fault(PcuExecutionFaultKind::InvalidFloatingOperand, 2, false),
            &mut status,
        );
        assert_eq!(
            check(
                &ordinary,
                &encode(&finite),
                PcuCompletionOutcome::Succeeded,
                &mut status
            ),
            expected
        );
    }
    assert_sequential_unary_lifecycle(
        backend,
        &clamp,
        value_type,
        &encode(&finite),
        &expected,
        &encode(&subnormal),
        &recovered,
    );
    // Encoding corpus includes lattice boundaries and deterministic finite random signs/exponents.
    let tiny_maximum = if width == 4 {
        0x007f_ffff
    } else {
        0x000f_ffff_ffff_ffff
    };
    let min_normal = tiny_maximum + 1;
    let mut corpus = vec![
        0,
        sign,
        1,
        sign | 1,
        tiny_maximum,
        tiny_maximum | sign,
        min_normal,
        min_normal | sign,
        maximum,
        maximum | sign,
        normal,
        normal | sign,
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..52 {
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9).wrapping_add(1);
        let bits = if width == 4 {
            state & u64::from(u32::MAX)
        } else {
            state
        };
        corpus.push(bits & (maximum | sign));
    }
    for chunk in corpus.as_chunks::<4>().0 {
        let wanted = encode(&chunk.iter().map(|bits| bits ^ sign).collect::<Vec<_>>());
        assert_eq!(
            check(
                &ordinary,
                &encode(chunk),
                PcuCompletionOutcome::Succeeded,
                &mut status
            ),
            wanted
        );
    }
}

#[test]
#[ignore = "requires a working ROCm device"]
fn checked_neg_direct_and_grid_cover_f32_f64_bits_faults_recovery_retry_and_corpus() {
    let (_discovery, backend) = super::checked_integer_tests::selected_device();
    for grid in [false, true] {
        for value_type in [PcuValueType::f32(), PcuValueType::f64()] {
            assert_neg_width(&backend, value_type, grid);
        }
    }
}

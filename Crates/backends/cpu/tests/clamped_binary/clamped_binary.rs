//! Observable completed Clamp, fatal priority and whole-map rollback for all six formats.
#[path = "source/source.rs"]
#[allow(dead_code)] // Ordinary calls become active after exact host Clamp offer admission.
mod source;
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedBinary,
    PcuCpuPreparedBinaryError,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchOp,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
#[path = "support/support.rs"]
mod support;
use support::Bits;
const OPS: [PcuDispatchFloatBinaryOp; 4] = [
    PcuDispatchFloatBinaryOp::Add,
    PcuDispatchFloatBinaryOp::Sub,
    PcuDispatchFloatBinaryOp::Mul,
    PcuDispatchFloatBinaryOp::Div,
];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
];
#[allow(clippy::too_many_lines)] // One six-format proof keeps recovered/fatal precedence, underflow payloads and retries together.
#[allow(clippy::cognitive_complexity)] // Complete format/policy/publication matrix preserves one owner lifecycle and exact case ordering.
fn width<T: Bits>() {
    let one = T::from_bits(T::ONE);
    let half = T::from_bits(T::ONE - T::MIN_NORMAL);
    let two = T::from_bits(T::ONE + T::MIN_NORMAL);
    let max = T::from_bits(T::MAX);
    let minsub = T::from_bits(1);
    let zero = T::from_bits(0);
    let negative_zero = T::from_bits(T::SIGN);
    let sentinel = T::from_bits(T::ONE + 1);
    let bindings = source::add_bindings::<T>();
    let builder = source::add_ir::<T, 7>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut operations = kernel.ops.to_vec();
    for op in OPS {
        for policy in POLICIES {
            for operation in &mut operations {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    op: selected,
                    underflow_policy,
                    ..
                }) = operation
                {
                    *selected = op;
                    *underflow_policy = policy;
                }
            }
            let kernel = pcu_facade::PcuDispatchKernelIr {
                ops: &operations,
                ..kernel
            };
            let mut prepared = PcuCpuCheckedBinary::<T>::new()
                .prepare_host_kernel(&kernel)
                .unwrap();
            let (range_left, range_right) = match op {
                PcuDispatchFloatBinaryOp::Add => (max, max),
                PcuDispatchFloatBinaryOp::Sub => (max, T::from_bits(T::SIGN | T::MAX)),
                PcuDispatchFloatBinaryOp::Mul => (max, two),
                PcuDispatchFloatBinaryOp::Div => (max, half),
            };
            let mut left = [one; 7];
            let mut right = [one; 7];
            left[1] = range_left;
            right[1] = range_right;
            left[5] = range_left;
            right[5] = range_right;
            let mut output = [sentinel; 9];
            let mut invoke = |left: &[T], right: &[T], output: &mut [T]| {
                prepared.call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), left),
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), right),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
                ])
            };
            let fault = invoke(&left, &right, &mut output).unwrap_err();
            assert!(
                matches!(fault, PcuCpuPreparedBinaryError::Fault(fault) if fault.recovered && fault.invocation_id == 1 && fault.kind == PcuExecutionFaultKind::ArithmeticOverflow)
            );
            assert_eq!(output[1].bits(), T::MAX);
            assert_eq!(output[5].bits(), T::MAX);
            assert_eq!(output[7..], [sentinel; 2]);
            let normal = match op {
                PcuDispatchFloatBinaryOp::Add => two,
                PcuDispatchFloatBinaryOp::Sub => zero,
                _ => one,
            };
            for index in [0, 2, 3, 4, 6] {
                assert_eq!(output[index], normal);
            }
            // Fatal-before/after-recovery and multiple fatal lanes never publish partial useful output.
            for fatal in [0, 3, 6] {
                let mut invalid = left;
                invalid[fatal] = T::from_bits(T::SIGN - 1);
                let before = output;
                let fault = invoke(&invalid, &right, &mut output).unwrap_err();
                assert!(
                    matches!(fault, PcuCpuPreparedBinaryError::Fault(fault) if !fault.recovered && fault.invocation_id == fatal as u64 && fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand)
                );
                assert_eq!(output, before);
                assert!(
                    matches!(invoke(&left, &right, &mut output), Err(PcuCpuPreparedBinaryError::Fault(fault)) if fault.recovered)
                );
            }
            let before = output;
            assert!(invoke(&left[..6], &right, &mut output).is_err());
            assert_eq!(output, before);
            if matches!(
                op,
                PcuDispatchFloatBinaryOp::Mul | PcuDispatchFloatBinaryOp::Div
            ) {
                let tiny_right = if op == PcuDispatchFloatBinaryOp::Mul {
                    half
                } else {
                    two
                };
                let result = invoke(&[minsub; 7], &[tiny_right; 7], &mut output);
                if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                    result.unwrap();
                } else {
                    assert!(
                        matches!(result, Err(PcuCpuPreparedBinaryError::Fault(fault)) if fault.recovered && fault.invocation_id == 0 && fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow)
                    );
                }
                assert_eq!(output[..7], [zero; 7]);
                let negative = invoke(
                    &[T::from_bits(T::SIGN | 1); 7],
                    &[tiny_right; 7],
                    &mut output,
                );
                if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                    negative.unwrap();
                } else {
                    assert!(
                        matches!(negative, Err(PcuCpuPreparedBinaryError::Fault(fault)) if fault.recovered)
                    );
                }
                assert_eq!(output[..7], [negative_zero; 7]);
            }
        }
    }
    // First recovered logical lane wins among kinds; any later fatal overrides it transactionally.
    let mut mixed = source::mul_prepare::<T, 3, _>(&PcuCpuCheckedBinary::<T>::new()).unwrap();
    for reversed in [false, true] {
        let mut left = [minsub, max, one];
        let mut right = [half, two, one];
        if reversed {
            left.swap(0, 1);
            right.swap(0, 1);
        }
        let mut output = [sentinel; 5];
        assert!(
            matches!(mixed(&left, &right, &mut output), Err(PcuCpuPreparedBinaryError::Fault(fault)) if fault.recovered && fault.invocation_id == 0 && fault.kind == if reversed { PcuExecutionFaultKind::ArithmeticOverflow } else { PcuExecutionFaultKind::ArithmeticUnderflow })
        );
        assert_eq!(
            output[..3],
            if reversed {
                [max, zero, one]
            } else {
                [zero, max, one]
            }
        );
        assert_eq!(output[3..], [sentinel; 2]);
        let before = output;
        left[2] = T::from_bits(T::SIGN - 1);
        assert!(
            matches!(mixed(&left, &right, &mut output), Err(PcuCpuPreparedBinaryError::Fault(fault)) if !fault.recovered && fault.invocation_id == 2)
        );
        assert_eq!(output, before);
    }
    let mut scale = source::scale_prepare::<T, 7, _>(&PcuCpuCheckedBinary::<T>::new()).unwrap();
    let mut output = [sentinel; 9];
    assert!(
        matches!(scale(&[minsub; 7], &one, &mut output), Err(PcuCpuPreparedBinaryError::Fault(fault)) if fault.recovered && fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(output[..7], [minsub; 7]);
    assert_eq!(output[7..], [sentinel; 2]);
    let mut mul = source::mul_prepare::<T, 7, _>(&PcuCpuCheckedBinary::<T>::new()).unwrap();
    assert!(
        matches!(mul(&[T::from_bits(T::MIN_NORMAL); 7], &[T::from_bits(T::ONE - 1); 7], &mut output), Err(PcuCpuPreparedBinaryError::Fault(fault)) if fault.recovered && fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(output[..7], [T::from_bits(T::MIN_NORMAL); 7]);
    let mut grid = source::grid_prepare::<T, 7, _>(&PcuCpuCheckedBinary::<T>::new()).unwrap();
    let left = [max, one, max, one, max, one, max];
    assert!(
        matches!(grid(&left, &[half; 7], &mut output), Err(PcuCpuPreparedBinaryError::Fault(fault)) if fault.recovered && fault.invocation_id == 0)
    );
    let before = output;
    let mut denominator = [half; 7];
    denominator[4] = zero;
    assert!(
        matches!(grid(&left, &denominator, &mut output), Err(PcuCpuPreparedBinaryError::Fault(fault)) if !fault.recovered && fault.invocation_id == 4 && fault.kind == PcuExecutionFaultKind::DivideByZero)
    );
    assert_eq!(output, before);
}
#[test]
fn f32_clamp() {
    width::<f32>();
}
#[test]
fn f64_clamp() {
    width::<f64>();
}
#[test]
fn f16_clamp() {
    width::<PcuF16Bits>();
}
#[test]
fn bf16_clamp() {
    width::<PcuBf16Bits>();
}
#[test]
fn e4m3fn_clamp() {
    width::<PcuF8E4M3FnBits>();
}
#[test]
fn e5m2_clamp() {
    width::<PcuF8E5M2Bits>();
}
#[cfg(feature = "source-clamp")]
fn ordinary<T: Bits>() {
    let one = T::from_bits(T::ONE);
    let half = T::from_bits(T::ONE - T::MIN_NORMAL);
    let max = T::from_bits(T::MAX);
    let zero = T::from_bits(0);
    let mut output = [one; 9];
    let left = [max, one, max, one, max, one, max];
    let fault = source::div::<T, 7>(&left, &[half; 7], &mut output).unwrap_err();
    assert!(
        matches!(fault, pcu_facade::global::PcuExecutionError::ArithmeticFault(fault) if fault.recovered && fault.invocation_id == 0 && fault.kind == PcuExecutionFaultKind::ArithmeticOverflow)
    );
    for index in [0, 2, 4, 6] {
        assert_eq!(output[index], max);
    }
    let before = output;
    let mut bad = [half; 7];
    bad[3] = zero;
    assert!(
        matches!(source::div::<T, 7>(&left, &bad, &mut output), Err(pcu_facade::global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id == 3 && fault.kind == PcuExecutionFaultKind::DivideByZero)
    );
    assert_eq!(output, before);
    assert!(
        matches!(source::div::<T, 7>(&left, &[half; 7], &mut output), Err(pcu_facade::global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert!(
        matches!(source::grid::<T, 7>(&left, &[half; 7], &mut output), Err(pcu_facade::global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert_eq!(output[7..], [one; 2]);
    assert!(
        matches!(source::scale::<T, 7>(&[T::from_bits(1); 7], &one, &mut output), Err(pcu_facade::global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered && fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(output[..7], [T::from_bits(1); 7]);
}
#[cfg(feature = "source-clamp")]
#[test]
fn genuine_ordinary_six_format_clamp_publication_and_fatal_rollback() {
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    ordinary::<f32>();
    ordinary::<f64>();
    ordinary::<PcuF16Bits>();
    ordinary::<PcuBf16Bits>();
    ordinary::<PcuF8E4M3FnBits>();
    ordinary::<PcuF8E5M2Bits>();
}
#[path = "../low_precision/oracle/oracle.rs"]
#[allow(dead_code)] // Clamp proof uses evaluate_clamped; Reject is qualified separately.
mod oracle;
fn exhaustive<T: oracle::Low>(all_pairs: bool) {
    let bindings = source::add_bindings::<T>();
    let builder = source::add_ir::<T, 1>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut operations = kernel.ops.to_vec();
    for op in OPS {
        for policy in POLICIES {
            for operation in &mut operations {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    op: selected,
                    underflow_policy,
                    ..
                }) = operation
                {
                    *selected = op;
                    *underflow_policy = policy;
                }
            }
            let kernel = pcu_facade::PcuDispatchKernelIr {
                ops: &operations,
                ..kernel
            };
            let mut prepared = PcuCpuCheckedBinary::<T>::new()
                .prepare_host_kernel(&kernel)
                .unwrap();
            let count = u32::from(T::FORMAT.sign) * 2;
            let edges = [
                0,
                T::FORMAT.sign,
                1,
                T::FORMAT.sign | 1,
                (1 << T::FORMAT.fraction) - 1,
                1 << T::FORMAT.fraction,
                T::FORMAT.max,
                T::FORMAT.max - 1,
                T::FORMAT.sign | T::FORMAT.max,
                T::FORMAT.max + 1,
                T::FORMAT.sign - 1,
            ];
            for left in 0..count {
                let left = u16::try_from(left).unwrap();
                let mut compare = |right: u16| {
                    let left_value = [T::from_bits(left)];
                    let right_value = [T::from_bits(right)];
                    let sentinel = T::from_bits(T::FORMAT.max);
                    let mut output = [sentinel; 3];
                    let result = prepared.call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &left_value),
                        PcuHostArgument::read(PcuBindingRef::new(0, 1), &right_value),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
                    ]);
                    match T::FORMAT.evaluate_clamped(left, right, op, policy) {
                        Ok((bits, fault)) => {
                            assert_eq!(
                                output[0].bits(),
                                bits,
                                "{op:?} {policy:?} {left:x} {right:x}"
                            );
                            match fault {
                                None => result.unwrap(),
                                Some(kind) => assert!(
                                    matches!(result, Err(PcuCpuPreparedBinaryError::Fault(fault)) if fault.recovered && fault.invocation_id == 0 && fault.kind == kind)
                                ),
                            }
                        }
                        Err(kind) => {
                            assert!(
                                matches!(result, Err(PcuCpuPreparedBinaryError::Fault(fault)) if !fault.recovered && fault.invocation_id == 0 && fault.kind == kind)
                            );
                            assert_eq!(output[0], sentinel);
                        }
                    }
                    assert_eq!(output[1..], [sentinel; 2]);
                };
                if all_pairs {
                    for right in 0..count {
                        compare(u16::try_from(right).unwrap());
                    }
                } else {
                    for right in edges {
                        compare(right);
                    }
                    compare(left.wrapping_mul(40503).wrapping_add(17));
                }
            }
        }
    }
}
#[test]
fn exhaustive_e4m3fn_clamp_oracle() {
    exhaustive::<PcuF8E4M3FnBits>(true);
}
#[test]
fn exhaustive_e5m2_clamp_oracle() {
    exhaustive::<PcuF8E5M2Bits>(true);
}
#[test]
fn all_f16_encoding_clamp_oracle() {
    exhaustive::<PcuF16Bits>(false);
}
#[test]
fn all_bf16_encoding_clamp_oracle() {
    exhaustive::<PcuBf16Bits>(false);
}
#[allow(clippy::too_many_lines)] // Exact six-format operation/policy/permission admission matrix, including negatives.
fn offers<T: Bits>(base: u32) {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuCompoundArithmeticPolicy,
        PcuCostBoundary,
        PcuDeviceIdentity,
        PcuExecutorId,
        PcuImplementationOffers,
        PcuImplementationRequest,
        PcuNumericalMode,
        PcuObjectKind,
        PcuObjectRef,
        PcuPrecisionPolicy,
        PcuProviderId,
        PcuRangePolicy,
        PcuReproducibility,
    };
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let backend = fusion_pcu_cpu::PcuCpuHostBackend::scalar();
    let offers = fusion_pcu_cpu::PcuCpuHostOffers::new(backend, device, PcuExecutorId(0));
    let bindings = source::add_bindings::<T>();
    let builder = source::add_ir::<T, 3>(&bindings).unwrap();
    let template = builder.ir();
    let mut operations = template.ops.to_vec();
    for (offset, op) in OPS.into_iter().enumerate() {
        for policy in POLICIES {
            for operation in &mut operations {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    op: selected,
                    underflow_policy,
                    ..
                }) = operation
                {
                    *selected = op;
                    *underflow_policy = policy;
                }
            }
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for compound in [
                    PcuCompoundArithmeticPolicy::Checked,
                    PcuCompoundArithmeticPolicy::BackendDefined,
                ] {
                    for precision in [
                        PcuPrecisionPolicy::Preserve,
                        PcuPrecisionPolicy::BackendOptimized,
                    ] {
                        let mut kernel = pcu_facade::PcuDispatchKernelIr {
                            ops: &operations,
                            ..template
                        };
                        kernel.numerical_requirements.numerical_mode = mode;
                        kernel.numerical_requirements.float_underflow = policy;
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .compound_arithmetic = compound;
                        kernel.numerical_requirements.numerical_options.precision = precision;
                        assert_eq!(
                            backend.prepare_host_kernel(&kernel).unwrap().range_policy(),
                            PcuRangePolicy::Clamp
                        );
                        let mut request = PcuImplementationRequest {
                            device,
                            executor: PcuExecutorId(0),
                            operation: &kernel,
                            requirements: kernel.numerical_requirements,
                            boundary: PcuCostBoundary::Host,
                        };
                        let mut output = [None];
                        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
                        let offer = output[0].unwrap();
                        assert_eq!(
                            (offer.implementation.local_id, offer.implementation.revision),
                            (base + u32::try_from(offset).unwrap(), 2)
                        );
                        assert_eq!(offer.requirements, kernel.numerical_requirements);
                        assert_eq!(offer.workspace_bytes, Some(0));
                        request.requirements.range_policy = PcuRangePolicy::Reject;
                        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                        request.requirements = kernel.numerical_requirements;
                        request.requirements.float_underflow =
                            if policy == PcuFloatUnderflowPolicy::IeeeAfterRounding {
                                PcuFloatUnderflowPolicy::AllowGradualUnderflow
                            } else {
                                PcuFloatUnderflowPolicy::IeeeAfterRounding
                            };
                        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                        request.requirements = kernel.numerical_requirements;
                        request.boundary = PcuCostBoundary::Resident;
                        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                        assert!(backend.prepare_host_kernel(&kernel).is_err());
                        assert!(
                            PcuCpuCheckedBinary::<T>::new()
                                .prepare_host_kernel(&kernel)
                                .is_err()
                        );
                        let request = PcuImplementationRequest {
                            device,
                            executor: PcuExecutorId(0),
                            operation: &kernel,
                            requirements: kernel.numerical_requirements,
                            boundary: PcuCostBoundary::Host,
                        };
                        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                    }
                }
            }
        }
    }
}
#[test]
fn exact_clamp_ids_and_independent_numerical_permissions() {
    offers::<f32>(192);
    offers::<f64>(196);
    offers::<PcuF16Bits>(200);
    offers::<PcuBf16Bits>(204);
    offers::<PcuF8E4M3FnBits>(208);
    offers::<PcuF8E5M2Bits>(212);
}

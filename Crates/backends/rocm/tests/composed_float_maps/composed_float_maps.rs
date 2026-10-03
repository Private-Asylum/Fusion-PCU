//! Genuine low-format composition under complete flags, policies and publication laws.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/composed_float_maps/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/composed_float_maps/source/source.rs"]
#[allow(dead_code)] // All source layouts also participate in the canonical paired bench.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCompoundArithmeticPolicy,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
};
use oracle::Format;
fn bits<T: Format>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.bits(), expected.bits());
    }
}
fn fault(
    result: Result<(), PcuExecutionError>,
    index: u64,
    kind: PcuExecutionFaultKind,
    recovered: bool,
) {
    let error = result.unwrap_err();
    let actual = error
        .arithmetic_fault()
        .unwrap_or_else(|| panic!("{error:?}"));
    assert_eq!(
        (actual.invocation_id, actual.kind, actual.recovered),
        (index, kind, recovered)
    );
}
fn call<T: Format>(kind: u32, input: &[T], output: &mut [T]) -> Result<(), PcuExecutionError> {
    match kind {
        0 => source::map::single::<T, 7>(output, input),
        1 => source::map::unused::<T, 7>(&[], output, input),
        2 => source::map::grid::<T, 7>(&[], output, input),
        3 => source::map::seed::<T, 7>(&[], output, &input[0]),
        _ => unreachable!(),
    }
}
#[allow(clippy::too_many_lines)] // Complete policy and disposition witnesses share one source bank.
fn format<T: Format>(policy: global::PcuExecutionPolicy) {
    global::clear_thread_cache().unwrap();
    let range = policy.range_policy;
    let uf = policy.float_underflow;
    for kind in 0..4 {
        let (input, expected) = oracle::inputs::<T>(if kind == 3 { 1 } else { 7 }, 1);
        let expected = if kind == 3 {
            vec![expected[0]; 7]
        } else {
            expected
        };
        let mut output = [T::sentinel(); 9];
        call(kind, &input, &mut output).unwrap();
        oracle::verify(&expected, &output);
        let before = output;
        let mut bad = input.clone();
        bad[if kind == 3 { 0 } else { 2 }] = T::from(T::MAX + 1);
        fault(
            call(kind, &bad, &mut output),
            if kind == 3 { 0 } else { 2 },
            PcuExecutionFaultKind::InvalidFloatingOperand,
            false,
        );
        bits(&output, &before);
        call(kind, &input, &mut output).unwrap();
        oracle::verify(&expected, &output);
    }
    let two = T::from(T::ONE + T::MIN_NORMAL);
    let mut input = [T::one(); 7];
    let mut output = [T::sentinel(); 9];
    call(1, &input, &mut output).unwrap();
    let before = output;
    input[2] = T::from(T::MAX);
    input[6] = T::from(T::MAX);
    fault(
        call(1, &input, &mut output),
        2,
        PcuExecutionFaultKind::ArithmeticOverflow,
        range == PcuRangePolicy::Clamp,
    );
    if range == PcuRangePolicy::Clamp {
        assert_eq!(output[2].bits(), T::MAX);
        assert_eq!(output[6].bits(), T::MAX);
    } else {
        bits(&output, &before);
    }
    let prior = output;
    input[0] = T::from(T::MAX);
    input[6] = T::from(T::MAX + 1);
    fault(
        call(2, &input, &mut output),
        if range == PcuRangePolicy::Clamp { 6 } else { 0 },
        if range == PcuRangePolicy::Clamp {
            PcuExecutionFaultKind::InvalidFloatingOperand
        } else {
            PcuExecutionFaultKind::ArithmeticOverflow
        },
        false,
    );
    bits(&output, &prior);
    input = [T::one(); 7];
    input[2] = T::from(1);
    let before = output;
    if uf == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
        call(1, &input, &mut output).unwrap();
    } else {
        fault(
            call(1, &input, &mut output),
            2,
            PcuExecutionFaultKind::ArithmeticUnderflow,
            range == PcuRangePolicy::Clamp,
        );
    }
    if uf == PcuFloatUnderflowPolicy::AllowGradualUnderflow || range == PcuRangePolicy::Clamp {
        assert_eq!(output[2].bits(), 0);
    } else {
        bits(&output, &before);
    }
    call(1, &[T::from(T::SIGN); 7], &mut output).unwrap();
    bits(&output[..7], &[T::zero(); 7]);
    call(1, &[T::one(); 7], &mut output).unwrap();
    bits(&output[..7], &[two; 7]);
    bits(&output[7..], &[T::sentinel(); 2]);
}
#[test]
#[ignore = "authorized actual selected GPU: four named formats composed source/full flags/all UF/Reject-Clamp"]
fn low_formats_composed_source_bits_faults_and_retry() {
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        let policy = global::PcuExecutionPolicy {
                            backend: global::PcuBackendChoice::Rocm,
                            device: Some(0),
                            numerical_mode,
                            float_underflow,
                            range_policy,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                ..Default::default()
                            },
                            ..Default::default()
                        };
                        global::configure(policy).unwrap();
                        format::<fusion_pcu::PcuF16Bits>(policy);
                        format::<fusion_pcu::PcuBf16Bits>(policy);
                        format::<fusion_pcu::PcuF8E4M3FnBits>(policy);
                        format::<fusion_pcu::PcuF8E5M2Bits>(policy);
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "authorized actual selected GPU: composed resident discard/unread owner and same-session retry"]
fn low_formats_composed_resident_publication_and_unread_owner() {
    fn format<T: Format>() {
        global::clear_thread_cache().unwrap();
        let input = source::identity(&[T::one(); 7]).unwrap();
        let mut bad = [T::one(); 7];
        bad[2] = T::from(T::MAX + 1);
        let bad = source::identity(&bad).unwrap();
        let mut output = source::identity(&[T::sentinel(); 9]).unwrap();
        let mut fresh = source::identity(&[T::sentinel(); 9]).unwrap();
        source::map::unused::<T, 7>(&[], &mut output, &input).unwrap();
        fault(
            source::map::unused::<T, 7>(&[], &mut output, &bad),
            2,
            PcuExecutionFaultKind::InvalidFloatingOperand,
            false,
        );
        let mut host = [T::sentinel(); 9];
        assert!(output.read_into(&mut host).is_err());
        bits(&host, &[T::sentinel(); 9]);
        source::map::unused::<T, 7>(&output, &mut fresh, &input).unwrap();
        fresh.read_into(&mut host).unwrap();
        bits(&host[..7], &[T::from(T::ONE + T::MIN_NORMAL); 7]);
        bits(&host[7..], &[T::sentinel(); 2]);
        let mut input_host = [T::zero(); 7];
        input.read_into(&mut input_host).unwrap();
        bits(&input_host, &[T::one(); 7]);
    }
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    })
    .unwrap();
    format::<fusion_pcu::PcuF16Bits>();
    format::<fusion_pcu::PcuBf16Bits>();
    format::<fusion_pcu::PcuF8E4M3FnBits>();
    format::<fusion_pcu::PcuF8E5M2Bits>();
}

#[path = "../../benches/composed_float_maps/native/native.rs"]
#[allow(dead_code)] // Host benchmark wrappers are separate from this private protocol probe.
mod native;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;

#[test]
#[ignore = "authorized actual GPU: independent low-format chain private bits/status/reset and guards"]
fn low_formats_composed_private_independent_bits_and_status() {
    fn format<T: Format>(backend: &fusion_pcu_rocm::RocmOwnedDispatchBackend) {
        let bindings = source::map::single_bindings::<T>();
        for uf in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let requirements = fusion_pcu::PcuImplementationRequirements {
                    float_underflow: uf,
                    range_policy: range,
                    ..fusion_pcu::PcuImplementationRequirements::DEFAULT
                };
                let builder = source::map::__single_ir_with_float_underflow_policy::<T, 1>(
                    &bindings,
                    uf,
                    range,
                    requirements,
                )
                .unwrap();
                let mut native = native::Native::new::<T, 1>(backend, &builder.ir(), 1);
                let mut previous = T::sentinel();
                for bits in oracle::independent::corpus::<T>() {
                    let input = T::from(bits);
                    let (word, result) = oracle::independent::expected(input, uf, range);
                    native.upload(0, &[input]);
                    assert_eq!(
                        native.submit(0),
                        word,
                        "{} input={bits:#x} {uf:?}/{range:?}",
                        T::LABEL
                    );
                    let mut output = [T::sentinel(); 3];
                    native.read_full(&mut output);
                    if let Some(value) = result {
                        previous = value;
                    }
                    assert_eq!(output[0].bits(), previous.bits());
                    bits_tail(&output[1..]);
                }
            }
        }
    }
    fn bits_tail<T: Format>(tail: &[T]) {
        bits(tail, &[T::sentinel(); 2]);
    }
    let (_, backend, _) = selection::selected_device();
    format::<fusion_pcu::PcuF16Bits>(&backend);
    format::<fusion_pcu::PcuBf16Bits>(&backend);
    format::<fusion_pcu::PcuF8E4M3FnBits>(&backend);
    format::<fusion_pcu::PcuF8E5M2Bits>(&backend);
}
#[test]
fn independent_composed_oracle_matches_checked_primitive_sequence() {
    fn core<T: Format + fusion_pcu::PcuClampedFloat>(
        input: T,
        uf: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
    ) -> (u64, Option<T>) {
        let code = |kind| match kind {
            PcuExecutionFaultKind::ArithmeticOverflow => 3,
            PcuExecutionFaultKind::ArithmeticUnderflow => 4,
            PcuExecutionFaultKind::InvalidFloatingOperand => 5,
            other => panic!("unexpected primitive fault {other:?}"),
        };
        let mut notice = 0;
        let sum = match input.pcu_clamped_add_with_policy(input, uf) {
            Ok(value) => value,
            Err(fusion_pcu::PcuClampedError::Range(error)) => {
                let tag = code(error.kind());
                if range == PcuRangePolicy::Reject {
                    return (tag, None);
                }
                notice = tag;
                error.clamped_value()
            }
            Err(fusion_pcu::PcuClampedError::Fatal(kind)) => return (code(kind), None),
        };
        let value = match sum.pcu_clamped_mul_with_policy(input, uf) {
            Ok(value) => value,
            Err(fusion_pcu::PcuClampedError::Range(error)) => {
                let tag = code(error.kind());
                if range == PcuRangePolicy::Reject {
                    return (tag, None);
                }
                if notice == 0 {
                    notice = tag;
                }
                error.clamped_value()
            }
            Err(fusion_pcu::PcuClampedError::Fatal(kind)) => return (code(kind), None),
        };
        (
            if notice == 0 {
                u64::MAX
            } else {
                (1 << 63) | notice
            },
            Some(value),
        )
    }
    fn format<T: Format + fusion_pcu::PcuClampedFloat>() {
        for uf in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for bits in oracle::independent::corpus::<T>() {
                    let input = T::from(bits);
                    let independent = oracle::independent::expected(input, uf, range);
                    let core = core(input, uf, range);
                    assert_eq!(
                        (independent.0, independent.1.map(Format::bits)),
                        (core.0, core.1.map(Format::bits)),
                        "{} input={bits:#x} {uf:?}/{range:?}",
                        T::LABEL
                    );
                }
            }
        }
    }
    format::<fusion_pcu::PcuF16Bits>();
    format::<fusion_pcu::PcuBf16Bits>();
    format::<fusion_pcu::PcuF8E4M3FnBits>();
    format::<fusion_pcu::PcuF8E5M2Bits>();
}

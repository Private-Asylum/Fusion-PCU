//! Requested Portable unary exact bits, publication and retained private-status reset.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/portable_unary/native/native.rs"]
#[allow(dead_code)] // Timing wrappers are outside this independent private witness.
mod native;
#[path = "../../benches/portable_unary/oracle/oracle.rs"]
#[allow(dead_code)] // Healthy timing banks are exercised by the canonical benchmark.
mod oracle;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/portable_unary/source/source.rs"]
#[allow(dead_code)] // Every source layout has separate paired/canonical coverage.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuNumericalOptions,
    PcuRangePolicy,
    PcuReproducibility,
};
use oracle::Format;
fn requirements(
    uf: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> PcuImplementationRequirements {
    PcuImplementationRequirements {
        float_underflow: uf,
        range_policy: range,
        numerical_options: PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..Default::default()
        },
        ..PcuImplementationRequirements::DEFAULT
    }
}
const fn policies() -> [PcuFloatUnderflowPolicy; 3] {
    [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ]
}
fn equal<T: Format>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.bits(), expected.bits());
    }
}
#[allow(clippy::needless_pass_by_value)] // Consume one completed result for protocol classification.
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
#[test]
#[ignore = "actual selected GPU: exhaustive low encodings and sampled F32/F64 unary bits/status/reset"]
fn portable_unary_private_encoding_and_status_reset() {
    fn format<T: Format, const N: usize>(backend: &fusion_pcu_rocm::RocmOwnedDispatchBackend) {
        let input = oracle::corpus::<T, N>();
        for uf in policies() {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let requirement = requirements(uf, range);
                let neg_bindings = source::neg_bindings::<T>();
                let neg = source::__neg_ir_with_float_underflow_policy::<T, N>(
                    &neg_bindings,
                    uf,
                    range,
                    requirement,
                )
                .unwrap();
                let relu_bindings = source::relu_bindings::<T>();
                let relu = source::__relu_ir_with_float_underflow_policy::<T, N>(
                    &relu_bindings,
                    uf,
                    range,
                    requirement,
                )
                .unwrap();
                for (operation, ir) in [(0, neg.ir()), (1, relu.ir())] {
                    let mut native = native::Native::new::<T, N>(backend, &ir, 1);
                    let mut expected_word = u64::MAX;
                    let mut expected = vec![T::sentinel(); N + 2];
                    for (lane, (&input, output)) in input.iter().zip(&mut expected).enumerate() {
                        let (word, value) = oracle::expected(input, operation, uf, range);
                        if word != u64::MAX {
                            // The established GPU private ABI reserves three tag bits;
                            // the independent scalar oracle returns only tag/disposition.
                            expected_word =
                                expected_word.min(word | (u64::try_from(lane).unwrap() << 3));
                        }
                        if let Some(value) = value {
                            *output = value;
                        }
                    }
                    native.upload(0, &input);
                    assert_eq!(
                        native.submit(0),
                        expected_word,
                        "{} op={operation} {uf:?}/{range:?}",
                        T::LABEL
                    );
                    let mut actual = vec![T::sentinel(); N + 2];
                    native.read_full(&mut actual);
                    equal(&actual, &expected);
                    let healthy = vec![T::one(); N];
                    native.upload(0, &healthy);
                    assert_eq!(
                        native.submit(0),
                        u64::MAX,
                        "fault status must reset before success"
                    );
                    assert_eq!(
                        native.submit(0),
                        u64::MAX,
                        "terminal success may reuse the sentinel"
                    );
                    native.read_full(&mut actual);
                    equal(
                        &actual[..N],
                        &vec![
                            T::from(if operation == 0 {
                                T::SIGN | T::ONE
                            } else {
                                T::ONE
                            });
                            N
                        ],
                    );
                    equal(&actual[N..], &[T::sentinel(); 2]);
                }
            }
        }
    }
    let (_, backend, _) = selection::selected_device();
    format::<fusion_pcu::PcuF16Bits, 65536>(&backend);
    format::<fusion_pcu::PcuBf16Bits, 65536>(&backend);
    format::<fusion_pcu::PcuF8E4M3FnBits, 256>(&backend);
    format::<fusion_pcu::PcuF8E5M2Bits, 256>(&backend);
    format::<f32, 4096>(&backend);
    format::<f64, 4096>(&backend);
}
fn configure(uf: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    let requirement = requirements(uf, range);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        numerical_options: requirement.numerical_options,
        float_underflow: uf,
        range_policy: range,
        ..Default::default()
    })
    .unwrap();
}
#[test]
#[ignore = "actual selected GPU: requested Portable host rollback, Clamp notice, grid/broadcast and retry"]
fn portable_unary_source_publication_and_retry() {
    fn format<T: Format>() {
        for uf in policies() {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                configure(uf, range);
                global::clear_thread_cache().unwrap();
                let (input, expected) = oracle::inputs::<T>(65, 1, 0, uf);
                let mut output = [T::sentinel(); 67];
                source::neg::<T, 65>(&mut output, &input).unwrap();
                oracle::verify(&expected, &output);
                source::grid::<T, 65>(&[], &mut output, &input).unwrap();
                oracle::verify(&expected, &output);
                source::broadcast::<T, 65>(&mut output, &[T::from(T::SIGN | T::ONE)]).unwrap();
                oracle::verify(&[T::zero(); 65], &output);
                let before = output;
                let mut bad = [T::one(); 65];
                bad[2] = T::from(T::MAX + 1);
                bad[6] = T::from(T::MAX + 1);
                fault(
                    source::neg::<T, 65>(&mut output, &bad),
                    2,
                    PcuExecutionFaultKind::InvalidFloatingOperand,
                    false,
                );
                equal(&output, &before);
                bad[0] = T::from(1);
                fault(
                    source::neg::<T, 65>(&mut output, &bad),
                    if uf == PcuFloatUnderflowPolicy::RejectSubnormalResult
                        && range == PcuRangePolicy::Reject
                    {
                        0
                    } else {
                        2
                    },
                    if uf == PcuFloatUnderflowPolicy::RejectSubnormalResult
                        && range == PcuRangePolicy::Reject
                    {
                        PcuExecutionFaultKind::ArithmeticUnderflow
                    } else {
                        PcuExecutionFaultKind::InvalidFloatingOperand
                    },
                    false,
                );
                equal(&output, &before);
                bad = [T::one(); 65];
                bad[2] = T::from(1);
                if uf == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                    fault(
                        source::neg::<T, 65>(&mut output, &bad),
                        2,
                        PcuExecutionFaultKind::ArithmeticUnderflow,
                        range == PcuRangePolicy::Clamp,
                    );
                    if range == PcuRangePolicy::Reject {
                        equal(&output, &before);
                    } else {
                        assert_eq!(output[2].bits(), T::SIGN | 1);
                    }
                } else {
                    source::neg::<T, 65>(&mut output, &bad).unwrap();
                }
                source::neg::<T, 65>(&mut output, &input).unwrap();
                oracle::verify(&expected, &output);
            }
        }
    }
    format::<fusion_pcu::PcuF16Bits>();
    format::<fusion_pcu::PcuBf16Bits>();
    format::<fusion_pcu::PcuF8E4M3FnBits>();
    format::<fusion_pcu::PcuF8E5M2Bits>();
    format::<f32>();
    format::<f64>();
}
#[test]
#[ignore = "actual selected GPU: requested Portable resident discard, ignored discarded owner and same-session retry"]
fn portable_unary_resident_discard_and_retry() {
    fn format<T: Format>() {
        // Owned identity transport is created under Reject/Unspecified; scalar invocation
        // then requests Portable+Clamp. This does not admit owned graph recovery.
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Rocm,
            device: Some(0),
            ..Default::default()
        })
        .unwrap();
        global::clear_thread_cache().unwrap();
        let input = source::identity(&[T::one(); 65]).unwrap();
        let mut bad = [T::one(); 65];
        bad[2] = T::from(T::MAX + 1);
        let bad = source::identity(&bad).unwrap();
        let mut output = source::identity(&[T::sentinel(); 67]).unwrap();
        let mut fresh = source::identity(&[T::sentinel(); 67]).unwrap();
        let tiny = source::identity(&[T::from(1); 65]).unwrap();
        configure(
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuRangePolicy::Clamp,
        );
        fault(
            source::neg::<T, 65>(&mut output, &bad),
            2,
            PcuExecutionFaultKind::InvalidFloatingOperand,
            false,
        );
        let mut host = [T::sentinel(); 67];
        assert!(output.read_into(&mut host).is_err());
        equal(&host, &[T::sentinel(); 67]);
        source::grid::<T, 65>(&output, &mut fresh, &input).unwrap();
        fresh.read_into(&mut host).unwrap();
        oracle::verify(&[T::from(T::SIGN | T::ONE); 65], &host);
        fault(
            source::neg::<T, 65>(&mut fresh, &tiny),
            0,
            PcuExecutionFaultKind::ArithmeticUnderflow,
            true,
        );
        fresh.read_into(&mut host).unwrap();
        oracle::verify(&[T::from(T::SIGN | 1); 65], &host);
        source::neg::<T, 65>(&mut fresh, &input).unwrap();
        fresh.read_into(&mut host).unwrap();
        oracle::verify(&[T::from(T::SIGN | T::ONE); 65], &host);
    }
    format::<fusion_pcu::PcuF16Bits>();
    format::<fusion_pcu::PcuBf16Bits>();
    format::<fusion_pcu::PcuF8E4M3FnBits>();
    format::<fusion_pcu::PcuF8E5M2Bits>();
    format::<f32>();
    format::<f64>();
}

#[test]
fn independent_unary_oracle_matches_core_scalar_law() {
    fn format<T: Format + fusion_pcu::PcuClampedFloat, const N: usize>() {
        for input in oracle::corpus::<T, N>() {
            for uf in policies() {
                for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    for operation in [0, 1] {
                        let primitive = if operation == 0 {
                            input.pcu_clamped_neg_with_policy(uf)
                        } else {
                            input.pcu_clamped_relu_with_policy(uf)
                        };
                        let result = match primitive {
                            Ok(value) => (u64::MAX, Some(value.bits())),
                            Err(fusion_pcu::PcuClampedError::Fatal(kind)) => {
                                assert_eq!(kind, PcuExecutionFaultKind::InvalidFloatingOperand);
                                (5, None)
                            }
                            Err(fusion_pcu::PcuClampedError::Range(error)) => {
                                assert_eq!(
                                    error.kind(),
                                    PcuExecutionFaultKind::ArithmeticUnderflow
                                );
                                if range == PcuRangePolicy::Reject {
                                    (4, None)
                                } else {
                                    ((1 << 63) | 4, Some(error.clamped_value().bits()))
                                }
                            }
                        };
                        let expected = oracle::expected(input, operation, uf, range);
                        assert_eq!(
                            result,
                            (expected.0, expected.1.map(Format::bits)),
                            "{} {input:?} op={operation} {uf:?}/{range:?}",
                            T::LABEL
                        );
                    }
                }
            }
        }
    }
    format::<fusion_pcu::PcuF16Bits, 65536>();
    format::<fusion_pcu::PcuBf16Bits, 65536>();
    format::<fusion_pcu::PcuF8E4M3FnBits, 256>();
    format::<fusion_pcu::PcuF8E5M2Bits, 256>();
    format::<f32, 4096>();
    format::<f64, 4096>();
}

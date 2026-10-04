//! Independent conversion packing, exact cold profiles and transactional recovered publication.
#[path = "../prepared_tensor/allocation/allocation.rs"]
mod allocation;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "../prepared_conversion/source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuHostBackend,PcuCpuPreparedHost,PcuCpuHostError};
#[rustfmt::skip]
use pcu_facade::{global,PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuFloatUnderflowPolicy as Policy,PcuNumericalMode,PcuExecutionFaultKind as Kind,PcuDispatchOp,PcuDispatchDataOp,PcuDispatchKernelIr,PcuImplementationRequirements,PcuRangePolicy};
// Numerical globals invalidate every thread's prepared cache. Serialize the
// tests that change them so unrelated policy changes cannot enter a warm census.
static POLICY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn configure(policy: Policy, mode: PcuNumericalMode) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        float_underflow: policy,
        numerical_mode: mode,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
fn same(a: &[f32], b: &[f32]) {
    assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
}
fn call(
    plan: &mut PcuCpuPreparedHost,
    input: &[f64],
    output: &mut [f32],
) -> Result<(), PcuCpuHostError> {
    plan.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
    ])
}
const fn kernel<'a>(
    original: PcuDispatchKernelIr<'a>,
    body: &'a [PcuDispatchOp<'a>],
    policy: Policy,
    mode: PcuNumericalMode,
) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        ops: body,
        numerical_requirements: PcuImplementationRequirements {
            range_policy: PcuRangePolicy::Clamp,
            float_underflow: policy,
            numerical_mode: mode,
            ..original.numerical_requirements
        },
        ..original
    }
}
#[test]
fn independent_full_bit_rounding_recovery_and_widening() {
    let backend = PcuCpuHostBackend::scalar();
    let mut cases = 0;
    for policy in [
        Policy::IeeeAfterRounding,
        Policy::RejectSubnormalResult,
        Policy::AllowGradualUnderflow,
    ] {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            let bindings = source::narrow_clamp_bindings();
            let builder = source::narrow_clamp_ir::<1>(&bindings).unwrap();
            let original = builder.ir();
            let mut ops = original.ops.to_vec();
            for op in &mut ops {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                    underflow_policy,
                    ..
                }) = op
                {
                    *underflow_policy = policy;
                }
            }
            let mut plan = backend
                .prepare_host_kernel(&kernel(original, &ops, policy, mode))
                .unwrap();
            let mut state = 0x9876_5432_abcd_1234_u64;
            let mut output = [99_f32; 3];
            let edges = [
                0,
                1,
                0x8000_0000_0000_0000,
                0x36a0_0000_0000_0000,
                0x3690_0000_0000_0000,
                0x3810_0000_0000_0000,
                0x47ef_ffff_e000_0000,
                0x47ef_ffff_f000_0000,
                0x3ff0_0000_1000_0000,
                0x3ff0_0000_3000_0000,
                u64::MAX,
            ];
            for i in 0..65_536 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let bits = if i < edges.len() { edges[i] } else { state };
                output.fill(99.0);
                let actual = call(&mut plan, &[f64::from_bits(bits)], &mut output);
                match oracle::narrow(bits, policy) {
                    Err(kind) => {
                        let fault = actual.unwrap_err().fault().unwrap();
                        assert_eq!(fault.kind, kind);
                        assert!(!fault.recovered);
                        same(&output, &[99.0; 3]);
                    }
                    Ok((want, notice)) => {
                        assert_eq!(output[0].to_bits(), want, "bits={bits:x} {policy:?}");
                        if let Some(kind) = notice {
                            let fault = actual.unwrap_err().fault().unwrap();
                            assert_eq!(fault.kind, kind);
                            assert!(fault.recovered);
                        } else {
                            actual.unwrap();
                        }
                        same(&output[1..], &[99.0; 2]);
                    }
                }
                cases += 1;
            }
        }
    }
    let mut widened = source::widen_clamp_prepare::<1, _>(&backend).unwrap();
    let mut state = 0x2345_6789_u32;
    let mut output = [99_f64; 3];
    for _ in 0..65_536 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        output.fill(99.0);
        match oracle::widen(state) {
            Ok(want) => {
                widened(&[f32::from_bits(state)], &mut output).unwrap();
                assert_eq!(output[0].to_bits(), want);
            }
            Err(kind) => {
                assert_eq!(
                    widened(&[f32::from_bits(state)], &mut output)
                        .unwrap_err()
                        .fault()
                        .unwrap()
                        .kind,
                    kind
                );
                assert_eq!(output.map(f64::to_bits), [99.0_f64; 3].map(f64::to_bits));
            }
        }
        assert_eq!(output[1..], [99.0; 2]);
        cases += 1;
    }
    println!("independent clamped conversion raw cases={cases}");
}
#[test]
fn source_recovered_fatal_precedence_preflight_tails_retry_and_zero_heap() {
    let _guard = POLICY_LOCK.lock().unwrap();
    for policy in [
        Policy::IeeeAfterRounding,
        Policy::RejectSubnormalResult,
        Policy::AllowGradualUnderflow,
    ] {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            configure(policy, mode);
            let mut input = [1_f64; 65];
            input[2] = f64::MAX;
            input[6] = -f64::MAX;
            let mut output = [99_f32; 68];
            for grid in [false, true] {
                let error = if grid {
                    source::grid_clamp(&input, &mut output)
                } else {
                    source::narrow_clamp::<65>(&input, &mut output)
                }
                .unwrap_err();
                let notice = error.arithmetic_fault().unwrap();
                assert_eq!(
                    (notice.invocation_id, notice.kind, notice.recovered),
                    (2, Kind::ArithmeticOverflow, true)
                );
                assert_eq!(output[2].to_bits(), f32::MAX.to_bits());
                assert_eq!(output[6].to_bits(), (-f32::MAX).to_bits());
                same(&output[65..], &[99.0; 3]);
            }
            for (a, b) in [(f64::MAX, f64::NAN), (f64::NAN, f64::MAX)] {
                input[2] = a;
                input[6] = b;
                let saved = output;
                let notice = source::narrow_clamp::<65>(&input, &mut output)
                    .unwrap_err()
                    .arithmetic_fault()
                    .unwrap();
                assert_eq!(
                    (notice.invocation_id, notice.kind, notice.recovered),
                    (
                        if a.is_nan() { 2 } else { 6 },
                        Kind::InvalidFloatingOperand,
                        false
                    )
                );
                same(&output, &saved);
            }
            input.fill(1.0);
            source::narrow_clamp::<65>(&input, &mut output).unwrap();
            same(&output[..65], &[1.0; 65]);
            let saved = output;
            assert!(source::narrow_clamp::<65>(&input[..64], &mut output).is_err());
            assert!(source::narrow_clamp::<65>(&input, &mut output[..64]).is_err());
            same(&output, &saved);
            let notice = source::broadcast_clamp::<65>(&f64::MAX, &mut output)
                .unwrap_err()
                .arithmetic_fault()
                .unwrap();
            assert_eq!((notice.invocation_id, notice.recovered), (0, true));
            same(&output[..65], &[f32::MAX; 65]);
            same(&output[65..], &[99.0; 3]);
            assert_eq!(
                allocation::count(|| {
                    for _ in 0..64 {
                        source::narrow_clamp::<65>(&input, &mut output).unwrap();
                    }
                }),
                0
            );
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[test]
#[allow(clippy::too_many_lines)] // The two schema-specific macro expansions keep each exact cold tuple and negative beside its offer.
fn exact_cast_ids_tuple_offers_and_rejected_header_portable_profiles() {
    use fusion_pcu_cpu::{PcuCpuHostOffers, PcuCpuHostOfferError};
    #[rustfmt::skip]
    use pcu_facade::{PcuDeviceIdentity,PcuObjectRef,PcuProviderId,PcuObjectKind,PcuExecutorId,PcuImplementationRequest,PcuImplementationOffers,PcuCostBoundary,PcuCompoundArithmeticPolicy as Compound,PcuPrecisionPolicy as Precision,PcuReproducibility};
    let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 1,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), identity, PcuExecutorId(0));
    let mut cases = 0;
    macro_rules! direction {
        ($bindings:ident,$ir:ident,$reject:expr,$clamp:expr) => {{
            let bindings = source::$bindings();
            let builder = source::$ir::<65>(&bindings).unwrap();
            let original = builder.ir();
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for policy in [
                    Policy::IeeeAfterRounding,
                    Policy::RejectSubnormalResult,
                    Policy::AllowGradualUnderflow,
                ] {
                    for compound in [Compound::Checked, Compound::BackendDefined] {
                        for precision in [Precision::Preserve, Precision::BackendOptimized] {
                            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                                let mut ops = original.ops.to_vec();
                                for op in &mut ops {
                                    if let PcuDispatchOp::Data(
                                        PcuDispatchDataOp::CheckedFloatConvert {
                                            underflow_policy,
                                            range_policy,
                                            ..
                                        },
                                    ) = op
                                    {
                                        *underflow_policy = policy;
                                        *range_policy = range;
                                    }
                                }
                                let mut requirements = original.numerical_requirements;
                                requirements.numerical_mode = mode;
                                requirements.float_underflow = policy;
                                requirements.range_policy = range;
                                requirements.numerical_options.compound_arithmetic = compound;
                                requirements.numerical_options.precision = precision;
                                let kernel = PcuDispatchKernelIr {
                                    ops: &ops,
                                    numerical_requirements: requirements,
                                    ..original
                                };
                                let request = PcuImplementationRequest {
                                    device: identity,
                                    executor: PcuExecutorId(0),
                                    operation: &kernel,
                                    requirements,
                                    boundary: PcuCostBoundary::Host,
                                };
                                let mut output = [None];
                                assert_eq!(
                                    offers.implementation_offers(&request, &mut output),
                                    Ok(1)
                                );
                                assert_eq!(
                                    output[0].unwrap().implementation.local_id,
                                    if range == PcuRangePolicy::Reject {
                                        $reject
                                    } else {
                                        $clamp
                                    }
                                );
                                assert_eq!(output[0].unwrap().implementation.revision, 1);
                                let mut portable = kernel;
                                portable
                                    .numerical_requirements
                                    .numerical_options
                                    .reproducibility = PcuReproducibility::PortableV1;
                                let request = PcuImplementationRequest {
                                    operation: &portable,
                                    requirements: portable.numerical_requirements,
                                    ..request
                                };
                                assert_eq!(
                                    offers.implementation_offers(&request, &mut output),
                                    Ok(0)
                                );
                                assert!(
                                    PcuCpuHostBackend::scalar()
                                        .prepare_host_kernel(&portable)
                                        .is_err()
                                );
                                let mut mismatch = kernel;
                                mismatch.numerical_requirements.float_underflow =
                                    if policy == Policy::IeeeAfterRounding {
                                        Policy::AllowGradualUnderflow
                                    } else {
                                        Policy::IeeeAfterRounding
                                    };
                                assert!(matches!(
                                    PcuCpuHostBackend::scalar().prepare_host_kernel(&mismatch),
                                    Err(PcuCpuHostError::HeaderUnderflowMismatch)
                                ));
                                let request = PcuImplementationRequest {
                                    operation: &mismatch,
                                    requirements: mismatch.numerical_requirements,
                                    ..request
                                };
                                assert_eq!(
                                    offers.implementation_offers(&request, &mut output),
                                    Err(PcuCpuHostOfferError::UnderflowMismatch)
                                );
                                mismatch.numerical_requirements.float_underflow = policy;
                                mismatch.numerical_requirements.range_policy =
                                    if range == PcuRangePolicy::Clamp {
                                        PcuRangePolicy::Reject
                                    } else {
                                        PcuRangePolicy::Clamp
                                    };
                                assert!(matches!(
                                    PcuCpuHostBackend::scalar().prepare_host_kernel(&mismatch),
                                    Err(PcuCpuHostError::UnsupportedProfile)
                                ));
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }};
    }
    direction!(narrow_bindings, narrow_ir, 41, 417);
    direction!(widen_bindings, widen_ir, 40, 416);
    assert_eq!(cases, 96);
}

#[test]
fn normal_stored_result_still_reports_tiny_unbounded_precision_notice() {
    let _guard = POLICY_LOCK.lock().unwrap();
    // IEEE7.5(a) checks the unbounded nearest-24-bit result, not final stored exponent.
    let samples = [
        0x380f_ffff_dfff_ffff,
        0x380f_ffff_e000_0000,
        0x380f_ffff_e800_0000,
        0x380f_ffff_efff_ffff,
        0x380f_ffff_f000_0000,
        0x380f_ffff_f800_0000,
        0x3810_0000_0000_0000,
    ];
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::AllowGradualUnderflow,
            Policy::RejectSubnormalResult,
        ] {
            configure(policy, mode);
            let bindings = source::narrow_clamp_bindings();
            let builder = source::narrow_clamp_ir::<1>(&bindings).unwrap();
            let original = builder.ir();
            let mut ops = original.ops.to_vec();
            for op in &mut ops {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                    underflow_policy,
                    ..
                }) = op
                {
                    *underflow_policy = policy;
                }
            }
            let kernel = kernel(original, &ops, policy, mode);
            let mut prepared = PcuCpuHostBackend::scalar()
                .prepare_host_kernel(&kernel)
                .unwrap();
            for bits in samples {
                for sign in [0, 1_u64 << 63] {
                    let input = [f64::from_bits(bits | sign)];
                    let (wanted, notice) = oracle::narrow(bits | sign, policy).unwrap();
                    let mut explicit = [17.0_f32; 3];
                    let actual = call(&mut prepared, &input, &mut explicit)
                        .err()
                        .map(|error| error.fault().unwrap());
                    assert_eq!(
                        actual,
                        notice.map(|kind| pcu_facade::PcuExecutionFault {
                            kind,
                            invocation_id: 0,
                            recovered: true
                        })
                    );
                    assert_eq!(
                        explicit.map(f32::to_bits),
                        [wanted, 17.0_f32.to_bits(), 17.0_f32.to_bits()]
                    );
                    let mut ordinary = [17.0_f32; 3];
                    let actual = source::narrow_clamp::<1>(&input, &mut ordinary)
                        .err()
                        .map(|error| error.arithmetic_fault().unwrap());
                    assert_eq!(
                        actual,
                        notice.map(|kind| pcu_facade::PcuExecutionFault {
                            kind,
                            invocation_id: 0,
                            recovered: true
                        })
                    );
                    assert_eq!(ordinary.map(f32::to_bits), explicit.map(f32::to_bits));
                }
            }
            let mut output = [17.0_f32; 3];
            call(&mut prepared, &[1.25], &mut output).unwrap();
            assert_eq!(
                output.map(f32::to_bits),
                [1.25_f32, 17.0, 17.0].map(f32::to_bits)
            );
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    assert_eq!(
        oracle::narrow(0x380f_ffff_e800_0000, Policy::IeeeAfterRounding).unwrap(),
        (0x0080_0000, Some(Kind::ArithmeticUnderflow))
    );
}

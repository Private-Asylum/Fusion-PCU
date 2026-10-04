//! Cold projected ABI; genuine source, prepared IR and same native kernel full-host work.
#[rustfmt::skip]
use std::{
    time::{
        Duration,
        Instant,
    },
    sync::atomic::{
        AtomicUsize,
        Ordering,
    },
};
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuExecutionError,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuOwnedDispatchBackend,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuFloatUnderflowPolicy,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuNumericalOptions,
    PcuDispatchOp,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
};
use fusion_pcu_rocm::RocmOwnedDispatchBackend;
#[rustfmt::skip]
use super::{
    native::Native,
    oracle::{
        self,
        Format,
    },
    source,
};
static RETIREMENT_WITNESS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn guard() {
    if !std::env::args().any(|a| a == "--test") {
        super::activity::guard();
    }
}
fn arithmetic(ir: &PcuDispatchKernelIr<'_>) {
    fn visit(
        ops: &[PcuDispatchOp<'_>],
        requirements: PcuImplementationRequirements,
        count: &mut usize,
    ) {
        for operation in ops {
            match operation {
                PcuDispatchOp::GridStrideLoop { body, .. } => visit(body, requirements, count),
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    op,
                    range_policy,
                    underflow_policy,
                    ..
                }) => {
                    assert_eq!(
                        *op,
                        if *count == 0 {
                            PcuDispatchFloatBinaryOp::Add
                        } else {
                            PcuDispatchFloatBinaryOp::Mul
                        }
                    );
                    assert_eq!(*range_policy, requirements.range_policy);
                    assert_eq!(*underflow_policy, requirements.float_underflow);
                    *count += 1;
                }
                _ => (),
            }
        }
    }
    let mut count = 0;
    visit(ir.ops, ir.numerical_requirements, &mut count);
    assert_eq!(
        count, 2,
        "actual helper source must emit exactly Add then Mul"
    );
}
#[allow(clippy::too_many_lines)] // One scope matches both observable outputs and complete API boundary.
#[allow(clippy::cognitive_complexity)] // Six fixed measured routes retain one shared fault/census scope.
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    kind: u32,
    ir: &PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    if kind != 0 && std::env::var_os("FUSION_PCU_HELPER_TIMING_CURATED").is_some() {
        return;
    }
    super::activity::foreign_owners();
    arithmetic(ir);
    let mut prepared = backend.prepare_host_kernel(ir).unwrap();
    let mut native = Native::new::<T, N>(backend, ir, 1);
    let banks = [oracle::bank::<T>(N, 0), oracle::bank::<T>(N, 1)];
    let mut minimum = super::raw::Raw::new::<T, N>(backend, ir, [&banks[0].0, &banks[1].0]);
    let mut independent =
        super::raw::Raw::handwritten::<T, N>(backend, ir, [&banks[0].0, &banks[1].0]);
    if !RETIREMENT_WITNESS.swap(true, Ordering::Relaxed) {
        independent.known_terminal_retirement_witness();
        independent = super::raw::Raw::handwritten::<T, N>(backend, ir, [&banks[0].0, &banks[1].0]);
    }
    #[cfg(feature = "allocation-census")]
    let raw_counter = minimum.counter();
    #[cfg(feature = "allocation-census")]
    let independent_counter = independent.counter();
    let mut stage = vec![T::sentinel(); N + 2];
    let mut output = stage.clone();
    let mut invalid = banks[0].0.clone();
    invalid[2] = T::from(T::MAX + 1);
    #[cfg(feature = "allocation-census")]
    {
        fusion_pcu_rocm::reset_rocm_api_census();
        let (result, counts) =
            super::allocations::measure(|| host(&invalid, &mut stage, &mut output));
        assert!(result.is_err());
        eprintln!(
            "cold/source-first-invalid-call/{:?}/{}/{kind}/{N}: alloc={} realloc={} frees={} requested_bytes={}; API={:?}",
            ir.numerical_requirements,
            T::LABEL,
            counts.alloc_calls,
            counts.realloc_calls,
            counts.dealloc_calls,
            counts.requested_bytes,
            fusion_pcu_rocm::rocm_api_census()
        );
    }
    #[cfg(not(feature = "allocation-census"))]
    assert!(host(&invalid, &mut stage, &mut output).is_err());
    assert!(
        stage
            .iter()
            .chain(&output)
            .all(|value| value.bits() == T::sentinel().bits())
    );
    assert!(
        prepared
            .call(&mut [
                PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut stage),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                PcuHostArgument::read(PcuBindingRef::new(0, 2), &invalid),
            ])
            .is_err()
    );
    assert!(
        stage
            .iter()
            .chain(&output)
            .all(|value| value.bits() == T::sentinel().bits())
    );
    native.upload(0, &invalid);
    assert_eq!(
        native.submit(0),
        (2 << 3) | 5,
        "first invalid operand lane/status"
    );
    // Exact same status ABI and range/underflow request, independently authored arithmetic.
    // Known quiescence precedes each status reset; no failed result becomes public.
    for encoding in [T::MAX + 1, T::MAX, 1, T::MIN_NORMAL, T::SIGN] {
        let mut exceptional = banks[0].0.clone();
        exceptional[2] = T::from(encoding);
        native.upload(0, &exceptional);
        let actual_status = independent.probe(&exceptional);
        assert_eq!(
            actual_status,
            native.submit(0),
            "handwritten native status mismatch for encoding {encoding:#x}"
        );
        if actual_status == u64::MAX || actual_status & (1 << 63) != 0 {
            let mut want_stage = vec![T::sentinel(); N + 2];
            let mut want_output = want_stage.clone();
            native.read(&mut want_stage, &mut want_output);
            independent.verify(&want_stage[..N], &want_output[..N]);
        }
    }
    // Earlier recoverable range status must not mask a later fatal invalid operand.
    let mut mixed = banks[0].0.clone();
    mixed[1] = T::from(T::MAX);
    mixed[5] = T::from(T::MAX + 1);
    native.upload(0, &mixed);
    assert_eq!(independent.probe(&mixed), native.submit(0));
    let mut explicit = |input: &[T], stage: &mut [T], output: &mut [T]| {
        prepared
            .call(&mut [
                PcuHostArgument::read_write(PcuBindingRef::new(0, 0), stage),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                PcuHostArgument::read(PcuBindingRef::new(0, 2), input),
            ])
            .unwrap();
    };
    for (bank, (input, want_stage, want_output)) in banks.iter().enumerate() {
        host(input, &mut stage, &mut output).unwrap();
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
        explicit(input, &mut stage, &mut output);
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
        native.host(input, &mut stage, &mut output);
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
        native.minimal_host(input, &mut stage, &mut output);
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
        minimum.call(bank);
        minimum.verify(want_stage, want_output);
        independent.call(bank);
        independent.verify(want_stage, want_output);
        native.read_full(&mut stage, &mut output);
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!(
        "{}/{:?}/{:?}/{:?}/{:?}/{:?}/{}/{kind}/{N}",
        super::WORKLOAD_LABEL,
        ir.numerical_requirements.numerical_mode,
        ir.numerical_requirements.numerical_options.precision,
        ir.numerical_requirements
            .numerical_options
            .compound_arithmetic,
        ir.numerical_requirements.float_underflow,
        ir.numerical_requirements.range_policy,
        T::LABEL
    ));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for route in [
        "actual_source",
        "prepared_ir",
        "native_matched_boundary",
        "native_safe_endpoint",
        "native_owned_minimum",
        "handwritten_native_owned",
    ] {
        super::activity::foreign_owners();
        let mut bank = 0;
        let mut call = || {
            bank ^= 1;
            let (input, want_stage, want_output) = &banks[bank];
            let started = Instant::now();
            match route {
                "actual_source" => host(input, &mut stage, &mut output).unwrap(),
                "prepared_ir" => explicit(input, &mut stage, &mut output),
                "native_matched_boundary" => native.host(input, &mut stage, &mut output),
                "native_safe_endpoint" => native.minimal_host(input, &mut stage, &mut output),
                "native_owned_minimum" => minimum.call(bank),
                _ => independent.call(bank),
            }
            let elapsed = started.elapsed();
            if route == "handwritten_native_owned" {
                independent.verify(want_stage, want_output);
            } else if route == "native_owned_minimum" {
                minimum.verify(want_stage, want_output);
            } else {
                oracle::verify(want_stage, &stage);
                oracle::verify(want_output, &output);
            }
            assert_eq!(
                SCORES.load(Ordering::Relaxed),
                scores,
                "warm source reranked"
            );
            elapsed
        };
        let _ = call();
        #[cfg(feature = "allocation-census")]
        {
            raw_counter.set(super::raw::Api::default());
            independent_counter.set(super::raw::Api::default());
            fusion_pcu_rocm::reset_rocm_api_census();
            let ((), counts) = super::allocations::measure(|| {
                for _ in 0..64 {
                    std::hint::black_box(call());
                }
            });
            let api = fusion_pcu_rocm::rocm_api_census();
            assert_eq!(api.symbol_resolutions, 0);
            assert_eq!(api.module_loads, 0);
            assert_eq!(api.allocations, 0);
            assert_eq!(api.frees, 0);
            assert_eq!(
                api.kernel_launches,
                if route == "handwritten_native_owned" {
                    0
                } else {
                    64
                }
            );
            if route == "handwritten_native_owned" {
                assert_eq!(
                    (
                        counts.alloc_calls,
                        counts.realloc_calls,
                        counts.dealloc_calls
                    ),
                    (0, 0, 0)
                );
                let direct = independent_counter.get();
                assert_eq!(
                    (
                        direct.host_to_device_copies,
                        direct.device_to_host_copies,
                        direct.launches,
                        direct.event_records,
                        direct.event_waits
                    ),
                    (64, 192, 64, 64, 64)
                );
                eprintln!(
                    "handwritten-native-sdk/{:?}/{:?}/{:?}/{:?}/{:?}/{}/{kind}/{N}/64-changing-calls: {direct:?}",
                    ir.numerical_requirements.numerical_mode,
                    ir.numerical_requirements.numerical_options.precision,
                    ir.numerical_requirements
                        .numerical_options
                        .compound_arithmetic,
                    ir.numerical_requirements.float_underflow,
                    ir.numerical_requirements.range_policy,
                    T::LABEL
                );
                assert_eq!(
                    (
                        api.event_creates,
                        api.event_records,
                        api.event_waits,
                        api.event_destroys
                    ),
                    (0, 0, 0, 0)
                );
            }
            eprintln!(
                "census/{:?}/{:?}/{:?}/{:?}/{:?}/{}/{kind}/{N}/{route}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; API={api:?}",
                ir.numerical_requirements.numerical_mode,
                ir.numerical_requirements.numerical_options.precision,
                ir.numerical_requirements
                    .numerical_options
                    .compound_arithmetic,
                ir.numerical_requirements.float_underflow,
                ir.numerical_requirements.range_policy,
                T::LABEL,
                counts.alloc_calls,
                counts.realloc_calls,
                counts.dealloc_calls,
                counts.requested_bytes
            );
            if route == "native_owned_minimum" {
                let raw = raw_counter.get();
                assert_eq!(raw.host_to_device_copies, 64);
                assert_eq!(raw.device_to_host_copies, 192);
                eprintln!(
                    "raw-native-sdk/{}/{:?}/{:?}/{:?}/{:?}/{:?}/{kind}/{N}/64-changing-calls: {:?}",
                    T::LABEL,
                    ir.numerical_requirements.numerical_mode,
                    ir.numerical_requirements.numerical_options.precision,
                    ir.numerical_requirements
                        .numerical_options
                        .compound_arithmetic,
                    ir.numerical_requirements.float_underflow,
                    ir.numerical_requirements.range_policy,
                    raw
                );
                assert_eq!(api.host_to_device_copies, 0);
                assert_eq!(api.device_to_host_copies, 0);
            }
        }
        #[cfg(not(feature = "allocation-census"))]
        group.bench_function(route, |bench| {
            bench.iter_custom(|iterations| (0..iterations).map(|_| call()).sum::<Duration>());
        });
    }
    group.finish();
}
fn width_single<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    requirements: PcuImplementationRequirements,
) {
    let bindings = source::single::direct_bindings();
    let builder = source::single::__direct_ir_with_float_underflow_policy::<N>(
        &bindings,
        requirements.float_underflow,
        requirements.range_policy,
        requirements,
    )
    .unwrap();
    assert_eq!(builder.ir().numerical_requirements, requirements);
    case::<f32, N>(
        criterion,
        backend,
        0,
        &builder.ir(),
        |input, stage, output| source::single::direct::<N>(stage, output, input),
    );
    let builder = source::single::__grid_ir_with_float_underflow_policy::<N>(
        &bindings,
        requirements.float_underflow,
        requirements.range_policy,
        requirements,
    )
    .unwrap();
    builder.with_ir(|ir| {
        assert_eq!(ir.numerical_requirements, requirements);
        case::<f32, N>(criterion, backend, 1, ir, |input, stage, output| {
            source::single::grid::<N>(stage, output, input)
        });
    });
}
fn width_double<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    requirements: PcuImplementationRequirements,
) {
    let bindings = source::double::direct_bindings();
    let builder = source::double::__direct_ir_with_float_underflow_policy::<N>(
        &bindings,
        requirements.float_underflow,
        requirements.range_policy,
        requirements,
    )
    .unwrap();
    assert_eq!(builder.ir().numerical_requirements, requirements);
    case::<f64, N>(
        criterion,
        backend,
        0,
        &builder.ir(),
        |input, stage, output| source::double::direct::<N>(stage, output, input),
    );
    let builder = source::double::__grid_ir_with_float_underflow_policy::<N>(
        &bindings,
        requirements.float_underflow,
        requirements.range_policy,
        requirements,
    )
    .unwrap();
    builder.with_ir(|ir| {
        assert_eq!(ir.numerical_requirements, requirements);
        case::<f64, N>(criterion, backend, 1, ir, |input, stage, output| {
            source::double::grid::<N>(stage, output, input)
        });
    });
}
pub fn run(criterion: &mut Criterion) {
    let curated = std::env::var_os("FUSION_PCU_HELPER_TIMING_CURATED").is_some();
    guard();
    let (_, backend, _) = super::selection::selected_device();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            for compound_arithmetic in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                if curated
                    && (mode != PcuNumericalMode::Boundary
                        || precision != PcuPrecisionPolicy::Preserve
                        || compound_arithmetic != PcuCompoundArithmeticPolicy::Checked)
                {
                    continue;
                }
                permissions(
                    criterion,
                    &backend,
                    mode,
                    PcuNumericalOptions {
                        precision,
                        compound_arithmetic,
                        ..PcuImplementationRequirements::DEFAULT.numerical_options
                    },
                );
            }
        }
    }
}
fn permissions(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    mode: PcuNumericalMode,
    numerical_options: PcuNumericalOptions,
) {
    for uf in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        for range in [
            fusion_pcu::PcuRangePolicy::Reject,
            fusion_pcu::PcuRangePolicy::Clamp,
        ] {
            if std::env::var_os("FUSION_PCU_HELPER_TIMING_CURATED").is_some()
                && (uf != PcuFloatUnderflowPolicy::IeeeAfterRounding
                    || range != fusion_pcu::PcuRangePolicy::Reject)
            {
                continue;
            }
            let requirements = PcuImplementationRequirements {
                numerical_mode: mode,
                numerical_options,
                float_underflow: uf,
                range_policy: range,
            };
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Rocm,
                device: Some(backend.device_identity().device_id()),
                numerical_mode: mode,
                numerical_options,
                float_underflow: uf,
                range_policy: range,
                score_invocation: Some(score),
                ..Default::default()
            })
            .unwrap();
            width_single::<65>(criterion, backend, requirements);
            width_single::<4096>(criterion, backend, requirements);
            width_double::<65>(criterion, backend, requirements);
            width_double::<4096>(criterion, backend, requirements);
        }
    }
}

//! Matched changing-input full-host/resident boundaries with complete outside-timing oracle.
#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
    rc::Rc,
    time::{
        Duration,
        Instant,
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
    PcuDeviceTensor,
    PcuHostArgument,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuNumericalMode,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmOwnedDispatchBackend,
    RocmOwnedTensorAssessor,
    HipRuntime,
};
#[rustfmt::skip]
use super::{
    native::Native,
    oracle::{
        self,
        Scalar,
    },
    source,
};

fn preflight<T: Scalar>() -> Result<(), Box<dyn Error>> {
    let one = T::from_units(1, 1);
    let zero = T::default();
    for error in [
        source::boundary(&[one], &[one]).unwrap_err(),
        source::portable(&[one], &[one]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"));
    }
    for error in [
        source::update(&[one], &[T::tiny()]).unwrap_err(),
        source::overflow(&[one], &[T::max()]).unwrap_err(),
        source::update(&[T::infinity()], &[one]).unwrap_err(),
        source::chain(&[T::infinity()], &[one]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("CompoundArithmeticFault"));
    }
    let output = source::gradual(&[one], &[T::tiny()])?;
    let mut actual = [zero];
    output.read_into(&mut actual)?;
    oracle::verify(&[one], &actual);
    assert!(source::tight(&[T::tiny()], &[zero]).is_err());
    let output = source::zero(&[one], &[one])?;
    output.read_into(&mut actual)?;
    oracle::verify(&[one], &actual);
    let _ = source::witness(&[one], &[one])?;
    let output = source::native_strict(&[one], &[one])?;
    output.read_into(&mut actual)?;
    oracle::verify(&[T::from_units(1, 2)], &actual);
    let output = source::update(&[one], &[one])?;
    output.read_into(&mut actual)?;
    oracle::verify(&[T::from_units(1, 2)], &actual);
    Ok(())
}

static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn invocation_score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}
fn device_score(_: &fusion_pcu::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}
fn score_count() -> usize {
    SCORES.load(std::sync::atomic::Ordering::Relaxed)
}

fn configure_profile(
    requirements: fusion_pcu::PcuImplementationRequirements,
) -> Result<(), Box<dyn Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        numerical_options: requirements.numerical_options,
        float_underflow: requirements.float_underflow,
        block_size: 256,
        score_device: device_score,
        score_invocation: Some(invocation_score),
        ..Default::default()
    })?;
    global::clear_thread_cache()?;
    Ok(())
}

const DEFAULT_STRICT: fusion_pcu::PcuImplementationRequirements =
    fusion_pcu::PcuImplementationRequirements {
        numerical_mode: PcuNumericalMode::Strict,
        ..fusion_pcu::PcuImplementationRequirements::DEFAULT
    };

fn permission_profiles() -> Vec<fusion_pcu::PcuImplementationRequirements> {
    let mut profiles = Vec::new();
    for compound_arithmetic in [
        fusion_pcu::PcuCompoundArithmeticPolicy::Checked,
        fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            fusion_pcu::PcuPrecisionPolicy::Preserve,
            fusion_pcu::PcuPrecisionPolicy::BackendOptimized,
        ] {
            for float_underflow in [
                fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                fusion_pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                profiles.push(fusion_pcu::PcuImplementationRequirements {
                    numerical_options: fusion_pcu::PcuNumericalOptions {
                        compound_arithmetic,
                        precision,
                        ..Default::default()
                    },
                    float_underflow,
                    ..DEFAULT_STRICT
                });
            }
        }
    }
    profiles
}

#[allow(clippy::too_many_lines)] // All matching source/graph/native physical boundaries remain together.
fn case<T: Scalar, const N: usize>(
    criterion: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    runtime: &HipRuntime,
    requirements: fusion_pcu::PcuImplementationRequirements,
) -> Result<(), Box<dyn Error>> {
    configure_profile(requirements)?;
    super::activity::guard();
    let pool = PcuMemoryPoolId(0x5353_4744);
    let assessor_root = RocmOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = assessor_root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    graph.set_numerical_options(requirements.numerical_options);
    let w_id = graph.input([N], T::TYPE)?;
    let g_id = graph.input([N], T::TYPE)?;
    let o_id = graph.sgd_update(w_id, g_id, 0.5)?;
    graph.set_value_float_underflow_policy(o_id, requirements.float_underflow)?;
    let cold = Instant::now();
    let mut native = Native::new::<T, N>(runtime, &graph, o_id)?;
    let native_cold = cold.elapsed();
    let cold = Instant::now();
    let borrowed = assessor.prepare_graph(&graph, o_id)?;
    let prewarm = assessor.prewarm_prepared_graph(&borrowed)?;
    drop(borrowed);
    let program = graph.into_selected_program(
        &[o_id],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = assessor.prepare_owned_program(program)?;
    let graph_cold = cold.elapsed();
    let banks = [oracle::inputs::<T, N>(0), oracle::inputs::<T, N>(1)];
    let cold = Instant::now();
    drop(source::update(&banks[0].0, &banks[0].1)?);
    let source_first = cold.elapsed();
    eprintln!(
        "cold/{}:{N}: native={native_cold:?}, graph+prewarm={graph_cold:?} {prewarm:?}; source first completed call={source_first:?}; Strict requested={requirements:?}, ratebits=3f000000, destination width, separate Mul/Sub, no portable-bit claim",
        T::LABEL
    );
    let source_w = [
        source::identity(&banks[0].0)?,
        source::identity(&banks[1].0)?,
    ];
    let source_g = [
        source::identity(&banks[0].1)?,
        source::identity(&banks[1].1)?,
    ];
    let mut raw_w = [
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, banks[0].0.as_slice())?)?,
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, banks[1].0.as_slice())?)?,
    ];
    let mut raw_g = [
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, banks[0].1.as_slice())?)?,
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, banks[1].1.as_slice())?)?,
    ];
    for (bank, (w, g, _)) in banks.iter().enumerate() {
        native.upload(bank, w.as_slice(), g.as_slice())?;
    }
    let (submission, completion) = native.phases()?;
    eprintln!(
        "native checker untimed phase witness/{}:{N}: submission={submission:?}, completion+status+release={completion:?}; one call, no confidence interval",
        T::LABEL
    );
    let mut memory = backend.memory_provider(pool);
    let mut observed = vec![T::default(); N];
    for host in [true, false] {
        if !host {
            // Full-host controls mutate bank zero. Restore both physical banks before
            // resident samples, including when Criterion filtered out the host routes.
            for (bank, (w, g, _)) in banks.iter().enumerate() {
                memory
                    .transfer_to(
                        raw_w[bank].resource_mut(),
                        0,
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), w.as_slice()).bytes(),
                    )
                    .unwrap();
                memory
                    .transfer_to(
                        raw_g[bank].resource_mut(),
                        0,
                        PcuHostArgument::read(PcuBindingRef::new(0, 1), g.as_slice()).bytes(),
                    )
                    .unwrap();
                native.upload(bank, w.as_slice(), g.as_slice())?;
            }
        }
        let boundary = if host { "full_host" } else { "resident" };
        let mut group = criterion.benchmark_group(format!(
            "rocm_strict_sgd/{}/{requirements:?}/{boundary}/{N}",
            T::LABEL
        ));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(u64::try_from(N)?));
        for route in [
            "actual_source",
            "explicit_graph",
            "native_same_ordered_checker",
        ] {
            let mut phase = 0usize;
            let mut call = || {
                phase = (phase + 1) % 2;
                let bank = if host { 0 } else { phase };
                let (w, g, expected) = &banks[phase];
                let start = Instant::now();
                let mut total = match route {
                    "actual_source" => {
                        let output = if host {
                            source::update(w, g).unwrap()
                        } else {
                            source::update::<T, N>(&source_w[bank], &source_g[bank]).unwrap()
                        };
                        if host {
                            output.read_into(&mut observed).unwrap();
                        }
                        let mut total = start.elapsed();
                        if !host {
                            output.read_into(&mut observed).unwrap();
                        }
                        oracle::verify(expected, &observed);
                        let release = Instant::now();
                        drop(black_box(output));
                        total += release.elapsed();
                        total
                    }
                    "explicit_graph" => {
                        if host {
                            memory
                                .transfer_to(
                                    raw_w[0].resource_mut(),
                                    0,
                                    PcuHostArgument::read(PcuBindingRef::new(0, 0), w.as_slice())
                                        .bytes(),
                                )
                                .unwrap();
                            memory
                                .transfer_to(
                                    raw_g[0].resource_mut(),
                                    0,
                                    PcuHostArgument::read(PcuBindingRef::new(0, 1), g.as_slice())
                                        .bytes(),
                                )
                                .unwrap();
                        }
                        let output = assessor
                            .execute_owned_program_outputs(
                                &prepared,
                                &[(w_id, &raw_w[bank]), (g_id, &raw_g[bank])],
                                pool,
                                &mut memory,
                            )
                            .unwrap();
                        if host {
                            backend
                                .download_buffer(pool, output[0].1.buffer(), &mut observed)
                                .unwrap();
                        }
                        let mut total = start.elapsed();
                        if !host {
                            backend
                                .download_buffer(pool, output[0].1.buffer(), &mut observed)
                                .unwrap();
                        }
                        oracle::verify(expected, &observed);
                        let release = Instant::now();
                        drop(black_box(output));
                        total += release.elapsed();
                        total
                    }
                    _ => {
                        if host {
                            native.upload(0, w.as_slice(), g.as_slice()).unwrap();
                        }
                        let (output, fault) = native.submit(bank).unwrap();
                        assert_eq!(fault, u64::MAX);
                        if host {
                            Native::read(&output, &mut observed).unwrap();
                        }
                        let mut total = start.elapsed();
                        if !host {
                            Native::read(&output, &mut observed).unwrap();
                        }
                        oracle::verify(expected, &observed);
                        let release = Instant::now();
                        drop(black_box(output));
                        total += release.elapsed();
                        total
                    }
                };
                // The returned duration includes release; complete oracle/readback occurs every call.
                black_box(&mut total);
                total
            };
            let _ = call();
            let scores = score_count();
            #[cfg(feature = "allocation-census")]
            {
                fusion_pcu_rocm::reset_rocm_api_census();
                let ((), census) = super::allocations::measure(|| {
                    for _ in 0..64 {
                        let _ = call();
                    }
                });
                let api = fusion_pcu_rocm::rocm_api_census();
                assert_eq!(api.kernel_launches, 64);
                assert_eq!(api.symbol_resolutions, 0);
                assert_eq!(api.module_loads, 0);
                eprintln!(
                    "census/{}/{requirements:?}/{boundary}/{N}/{route}/64-changing-calls: Rust alloc={} realloc={} frees={} bytes={}; API={api:?}; resident API/Rust also includes untimed readback/oracle",
                    T::LABEL,
                    census.alloc_calls,
                    census.realloc_calls,
                    census.dealloc_calls,
                    census.requested_bytes
                );
            }
            #[cfg(feature = "allocation-census")]
            eprintln!(
                "selection_census/{}/{requirements:?}/{boundary}/{N}/{route}: before={scores} after={} calls=64",
                T::LABEL,
                score_count()
            );
            #[cfg(not(feature = "allocation-census"))]
            group.bench_function(route, |bench| {
                bench.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iterations {
                        total += call();
                    }
                    total
                });
            });
            assert_eq!(
                score_count(),
                scores,
                "warm source/graph/native calls must not rescore"
            );
        }
        group.finish();
    }
    Ok(())
}
pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    super::activity::guard();
    let (_discovery, backend, runtime) = super::selection::selected_device();
    super::correctness::run::<f32>(&backend, &runtime, DEFAULT_STRICT.numerical_options)?;
    super::correctness::run::<f64>(&backend, &runtime, DEFAULT_STRICT.numerical_options)?;
    preflight::<f32>()?;
    preflight::<f64>()?;
    case::<f32, 65>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f64, 65>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f32, 65_536>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f64, 65_536>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f32, 1_048_576>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f64, 1_048_576>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    for requirements in permission_profiles() {
        if requirements != DEFAULT_STRICT {
            if requirements.float_underflow
                == fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding
            {
                configure_profile(requirements)?;
                super::correctness::run::<f32>(&backend, &runtime, requirements.numerical_options)?;
                super::correctness::run::<f64>(&backend, &runtime, requirements.numerical_options)?;
            }
            case::<f32, 65>(criterion, &backend, &runtime, requirements)?;
            case::<f64, 65>(criterion, &backend, &runtime, requirements)?;
        }
    }
    global::use_defaults()?;
    global::clear_thread_cache()?;
    Ok(())
}

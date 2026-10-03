//! Setup and timing boundaries for strict source, graph and generated native arithmetic.
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
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuHostArgument,
    PcuBindingRef,
    PcuNumericalMode,
    global,
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
fn case<T: Scalar, const R: usize, const K: usize, const C: usize>(
    criterion: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    runtime: &HipRuntime,
    requirements: fusion_pcu::PcuImplementationRequirements,
) -> Result<(), Box<dyn Error>> {
    configure_profile(requirements)?;
    if std::env::var_os("FUSION_PCU_HELPER_TIMING_CURATED").is_some() {
        super::cohort_activity::foreign_owners();
    } else {
        super::activity::guard();
    }
    let pool = PcuMemoryPoolId(0);
    let assessor_root = RocmOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = assessor_root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    graph.set_numerical_options(requirements.numerical_options);
    let left_id = graph.input([R, K], T::TYPE)?;
    let right_id = graph.input([K, C], T::TYPE)?;
    let output_id = graph.matmul(left_id, right_id)?;
    graph.set_value_float_underflow_policy(output_id, requirements.float_underflow)?;
    let mut native = Native::new::<T, R, K, C>(runtime, &graph, output_id)?;
    let borrowed = assessor.prepare_graph(&graph, output_id)?;
    assessor.prewarm_prepared_graph(&borrowed)?;
    drop(borrowed);
    let program = graph.into_selected_program(
        &[output_id],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = assessor.prepare_owned_program(program)?;
    let mut memory = backend.memory_provider(pool);
    let mut observed = vec![T::default(); R * C];
    let (left, right) = oracle::fill::<T, R, K, C>(0);
    // Complete the source's first host operation before any measurement.
    drop(source::product::<T, R, K, C>(&left, &right)?);
    if requirements.numerical_options.compound_arithmetic
        == fusion_pcu::PcuCompoundArithmeticPolicy::Checked
    {
        let Err(boundary_error) = source::boundary::<T, R, K, C>(&left, &right) else {
            panic!("boundary mode must remain unadmitted");
        };
        assert!(
            super::correctness::boundary_rejected(&boundary_error),
            "unexpected boundary admission error: {boundary_error:?}"
        );
    }
    eprintln!(
        "{} {R}x{K}x{C}: Strict requested={requirements:?}; checked Boundary refusal tested where applicable; ordered separate multiply/add; block=256; source/graph/native fresh outputs and private fault status; vendor preallocated output; vendor BLAS different unchecked semantics",
        T::LABEL
    );
    #[cfg(feature = "allocation-census")]
    if cfg!(feature = "allocation-census") {
        let banks = [oracle::fill::<T, R, K, C>(0), oracle::fill::<T, R, K, C>(1)];
        let expected = banks
            .each_ref()
            .map(|(left, right)| oracle::expected(left, right));
        let source_left = [
            source::identity::<T, R, K>(&banks[0].0)?,
            source::identity::<T, R, K>(&banks[1].0)?,
        ];
        let source_right = [
            source::identity::<T, K, C>(&banks[0].1)?,
            source::identity::<T, K, C>(&banks[1].1)?,
        ];
        let raw_left = [
            PcuDeviceTensor::new(
                [R, K],
                backend.upload_buffer(pool, banks[0].0.as_flattened())?,
            )?,
            PcuDeviceTensor::new(
                [R, K],
                backend.upload_buffer(pool, banks[1].0.as_flattened())?,
            )?,
        ];
        let raw_right = [
            PcuDeviceTensor::new(
                [K, C],
                backend.upload_buffer(pool, banks[0].1.as_flattened())?,
            )?,
            PcuDeviceTensor::new(
                [K, C],
                backend.upload_buffer(pool, banks[1].1.as_flattened())?,
            )?,
        ];
        native.upload(banks[0].0.as_flattened(), banks[0].1.as_flattened())?;
        native.upload_bank(1, banks[1].0.as_flattened(), banks[1].1.as_flattened())?;
        for route in [
            "source_strict",
            "explicit_graph_strict",
            "native_same_ordered_checker_fresh",
        ] {
            let mut phase = 0usize;
            let mut call = || {
                phase = (phase + 1) % 2;
                match route {
                    "source_strict" => {
                        let output = source::product::<T, R, K, C>(
                            &source_left[phase],
                            &source_right[phase],
                        )
                        .unwrap();
                        output.read_into(&mut observed).unwrap();
                        drop(output);
                    }
                    "explicit_graph_strict" => {
                        let output = assessor
                            .execute_owned_program_outputs(
                                &prepared,
                                &[(left_id, &raw_left[phase]), (right_id, &raw_right[phase])],
                                pool,
                                &mut memory,
                            )
                            .unwrap();
                        backend
                            .download_buffer(pool, output[0].1.buffer(), &mut observed)
                            .unwrap();
                        drop(output);
                    }
                    _ => {
                        let (output, fault) = native.submit_fresh(phase).unwrap();
                        assert_eq!(fault, u64::MAX);
                        Native::read_fresh(&output, &mut observed).unwrap();
                        drop(output);
                    }
                }
                oracle::verify(&expected[phase], &observed);
            };
            call();
            let scores = score_count();
            fusion_pcu_rocm::reset_rocm_api_census();
            let ((), counts) = super::allocations::measure(|| {
                for _ in 0..64 {
                    call();
                }
            });
            let api = fusion_pcu_rocm::rocm_api_census();
            assert_eq!(api.kernel_launches, 64);
            assert_eq!(api.symbol_resolutions, 0);
            assert_eq!(api.module_loads, 0);
            assert_eq!(
                score_count(),
                scores,
                "warm checked routes must not rescore"
            );
            eprintln!(
                "selection_census/{}/{requirements:?}/{R}x{K}x{C}/{route}: before={scores} after={} calls=64",
                T::LABEL,
                score_count()
            );
            eprintln!(
                "census/{}/{requirements:?}/{R}x{K}x{C}/resident_readback/{route}/64-changing-calls: Rust alloc={} realloc={} frees={} bytes={}; API={api:?}; includes matched explicit readback/oracle",
                T::LABEL,
                counts.alloc_calls,
                counts.realloc_calls,
                counts.dealloc_calls,
                counts.requested_bytes
            );
        }
        return Ok(());
    }
    for host in [true, false] {
        let boundary = if host {
            "full_host"
        } else {
            "resident_fresh_output"
        };
        let (other_left, other_right) = oracle::fill::<T, R, K, C>(1);
        let source_left = [
            source::identity::<T, R, K>(&left)?,
            source::identity::<T, R, K>(&other_left)?,
        ];
        let source_right = [
            source::identity::<T, K, C>(&right)?,
            source::identity::<T, K, C>(&other_right)?,
        ];
        let mut raw_left = [
            PcuDeviceTensor::new([R, K], backend.upload_buffer(pool, left.as_flattened())?)?,
            PcuDeviceTensor::new(
                [R, K],
                backend.upload_buffer(pool, other_left.as_flattened())?,
            )?,
        ];
        let mut raw_right = [
            PcuDeviceTensor::new([K, C], backend.upload_buffer(pool, right.as_flattened())?)?,
            PcuDeviceTensor::new(
                [K, C],
                backend.upload_buffer(pool, other_right.as_flattened())?,
            )?,
        ];
        native.upload(left.as_flattened(), right.as_flattened())?;
        native.upload_bank(1, other_left.as_flattened(), other_right.as_flattened())?;
        let resident_expected = [
            oracle::expected(&left, &right),
            oracle::expected(&other_left, &other_right),
        ];
        if !host && !std::env::args().any(|arg| arg == "--test") {
            let orders = [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ];
            let mut ratios = [[0.0_f64; 36]; 2];
            for sample in 0..36 {
                let bank = sample & 1;
                let mut times = [Duration::ZERO; 3];
                for route in orders[sample % orders.len()] {
                    let start = Instant::now();
                    match route {
                        0 => {
                            let output = source::product::<T, R, K, C>(
                                &source_left[bank],
                                &source_right[bank],
                            )?;
                            times[route] = start.elapsed();
                            output.read_into(&mut observed)?;
                            oracle::verify(&resident_expected[bank], &observed);
                            let drop_start = Instant::now();
                            drop(output);
                            times[route] += drop_start.elapsed();
                        }
                        1 => {
                            let output = assessor.execute_owned_program_outputs(
                                &prepared,
                                &[(left_id, &raw_left[bank]), (right_id, &raw_right[bank])],
                                pool,
                                &mut memory,
                            )?;
                            times[route] = start.elapsed();
                            backend.download_buffer(pool, output[0].1.buffer(), &mut observed)?;
                            oracle::verify(&resident_expected[bank], &observed);
                            let drop_start = Instant::now();
                            drop(output);
                            times[route] += drop_start.elapsed();
                        }
                        _ => {
                            let (output, fault) = native.submit_fresh(bank)?;
                            assert_eq!(fault, u64::MAX);
                            times[route] = start.elapsed();
                            Native::read_fresh(&output, &mut observed)?;
                            oracle::verify(&resident_expected[bank], &observed);
                            let drop_start = Instant::now();
                            drop(output);
                            times[route] += drop_start.elapsed();
                        }
                    }
                }
                for route in 0..2 {
                    ratios[route][sample] = times[route].as_secs_f64() / times[2].as_secs_f64();
                }
            }
            for (label, mut samples) in ["source/native", "graph/native"].into_iter().zip(ratios) {
                samples.sort_by(f64::total_cmp);
                let median = samples[17].midpoint(samples[18]);
                eprintln!(
                    "paired/resident/{}/{R}x{K}x{C}/{label}: median={median:.6}; 36 interleaved triples, six orders equally repeated; actual ratios, no confidence interval or isolated-cost attribution",
                    T::LABEL
                );
            }
        }
        // Resident carrier preparation also precedes timing, independently of the host case.
        if !host {
            drop(source::product::<T, R, K, C>(
                &source_left[0],
                &source_right[0],
            )?);
        }
        let scores = score_count();
        let mut group = criterion.benchmark_group(format!(
            "rocm_strict_matmul/{}/{requirements:?}/{boundary}/{R}x{K}x{C}",
            T::LABEL
        ));
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(u64::try_from(R * C)?));
        for route in [
            "source_strict",
            "explicit_graph_strict",
            "native_same_ordered_checker_fresh",
            "vendor_blas_unchecked_different_semantics",
        ] {
            let mut job = 0_u64;
            group.bench_function(route, |bench| {
                bench.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iterations {
                        job = job.wrapping_add(1);
                        let (left, right) = oracle::fill::<T, R, K, C>(job);
                        let bank = if host {
                            0
                        } else {
                            usize::try_from(job & 1).unwrap()
                        };
                        let expected = if host {
                            oracle::expected(&left, &right)
                        } else {
                            resident_expected[bank].clone()
                        };
                        let start = Instant::now();
                        match route {
                            "source_strict" => {
                                let output = if host {
                                    source::product::<T, R, K, C>(&left, &right).unwrap()
                                } else {
                                    source::product::<T, R, K, C>(
                                        &source_left[bank],
                                        &source_right[bank],
                                    )
                                    .unwrap()
                                };
                                if host {
                                    output.read_into(&mut observed).unwrap();
                                }
                                total += start.elapsed();
                                if !host {
                                    output.read_into(&mut observed).unwrap();
                                }
                                oracle::verify(&expected, &observed);
                                let drop_start = Instant::now();
                                drop(black_box(output));
                                total += drop_start.elapsed();
                            }
                            "explicit_graph_strict" => {
                                if host {
                                    memory
                                        .transfer_to(
                                            raw_left[0].resource_mut(),
                                            0,
                                            PcuHostArgument::read(
                                                PcuBindingRef::new(0, 0),
                                                left.as_flattened(),
                                            )
                                            .bytes(),
                                        )
                                        .unwrap();
                                    memory
                                        .transfer_to(
                                            raw_right[0].resource_mut(),
                                            0,
                                            PcuHostArgument::read(
                                                PcuBindingRef::new(0, 1),
                                                right.as_flattened(),
                                            )
                                            .bytes(),
                                        )
                                        .unwrap();
                                }
                                let inputs =
                                    [(left_id, &raw_left[bank]), (right_id, &raw_right[bank])];
                                let output = assessor
                                    .execute_owned_program_outputs(
                                        &prepared,
                                        &inputs,
                                        pool,
                                        &mut memory,
                                    )
                                    .unwrap();
                                if host {
                                    backend
                                        .download_buffer(pool, output[0].1.buffer(), &mut observed)
                                        .unwrap();
                                }
                                total += start.elapsed();
                                if !host {
                                    backend
                                        .download_buffer(pool, output[0].1.buffer(), &mut observed)
                                        .unwrap();
                                }
                                oracle::verify(&expected, &observed);
                                let drop_start = Instant::now();
                                drop(black_box(output));
                                total += drop_start.elapsed();
                            }
                            _ => {
                                if host {
                                    native
                                        .upload(left.as_flattened(), right.as_flattened())
                                        .unwrap();
                                }
                                if route == "native_same_ordered_checker_fresh" {
                                    let (output, fault) = native.submit_fresh(bank).unwrap();
                                    assert_eq!(fault, u64::MAX);
                                    if host {
                                        Native::read_fresh(&output, &mut observed).unwrap();
                                    }
                                    total += start.elapsed();
                                    if !host {
                                        Native::read_fresh(&output, &mut observed).unwrap();
                                    }
                                    oracle::verify(&expected, &observed);
                                    let drop_start = Instant::now();
                                    drop(black_box(output));
                                    total += drop_start.elapsed();
                                } else {
                                    native.vendor::<T, R, K, C>(bank).unwrap();
                                    if host {
                                        native.read(&mut observed).unwrap();
                                    }
                                    total += start.elapsed();
                                    if !host {
                                        native.read(&mut observed).unwrap();
                                    }
                                    oracle::verify(&expected, &observed);
                                }
                            }
                        }
                    }
                    total
                });
            });
        }
        assert_eq!(
            score_count(),
            scores,
            "warm source/graph/native calls must not rescore"
        );
        group.finish();
    }
    Ok(())
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    if std::env::var_os("FUSION_PCU_HELPER_TIMING_CURATED").is_some() {
        super::cohort_activity::guard();
    } else {
        super::activity::guard();
    }
    let (_discovery, backend, runtime) = super::selection::selected_device();
    if std::env::var_os("FUSION_PCU_HELPER_TIMING_CURATED").is_some() {
        super::correctness::run::<f32>(&backend, &runtime, DEFAULT_STRICT.numerical_options)?;
        case::<f32, 4, 2, 4>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
        case::<f32, 64, 256, 64>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
        global::use_defaults()?;
        global::clear_thread_cache()?;
        return Ok(());
    }
    super::correctness::run::<f32>(&backend, &runtime, DEFAULT_STRICT.numerical_options)?;
    super::correctness::run::<f64>(&backend, &runtime, DEFAULT_STRICT.numerical_options)?;
    case::<f32, 4, 2, 4>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f64, 4, 2, 4>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f32, 32, 32, 32>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f64, 32, 32, 32>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f32, 64, 256, 64>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    case::<f64, 64, 256, 64>(criterion, &backend, &runtime, DEFAULT_STRICT)?;
    for requirements in permission_profiles() {
        if requirements != DEFAULT_STRICT {
            if requirements.float_underflow
                == fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding
            {
                configure_profile(requirements)?;
                super::correctness::run::<f32>(&backend, &runtime, requirements.numerical_options)?;
                super::correctness::run::<f64>(&backend, &runtime, requirements.numerical_options)?;
            }
            case::<f32, 4, 2, 4>(criterion, &backend, &runtime, requirements)?;
            case::<f64, 4, 2, 4>(criterion, &backend, &runtime, requirements)?;
        }
    }
    global::use_defaults()?;
    global::clear_thread_cache()?;
    Ok(())
}

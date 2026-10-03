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
    PcuCompoundArithmeticPolicy,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaOwnedDispatchBackend,
    CudaOwnedTensorAssessor,
    CudaRuntime,
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

#[allow(clippy::too_many_lines)] // Matching each source/graph/native resource boundary stays reviewable together.
fn case<T: Scalar, const N: usize>(
    criterion: &mut Criterion,
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
    optimized: bool,
) -> Result<(), Box<dyn Error>> {
    super::activity::activity_guard();
    let pool = PcuMemoryPoolId(0x5353_4744);
    let assessor_root = CudaOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = assessor_root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: if optimized {
            PcuPrecisionPolicy::BackendOptimized
        } else {
            PcuPrecisionPolicy::Preserve
        },
        ..PcuNumericalOptions::default()
    });
    let w_id = graph.input([N], T::TYPE)?;
    let g_id = graph.input([N], T::TYPE)?;
    let o_id = graph.mean_squared_error(w_id, g_id)?;
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
    drop(source::loss(&banks[0].0, &banks[0].1, optimized)?);
    let source_first = cold.elapsed();
    eprintln!(
        "cold/{}:{N}: native={native_cold:?}, graph+prewarm={graph_cold:?} {prewarm:?}; source first completed call={source_first:?}; Boundary native F64, rounded F64 difference/square + typed ASUM + F64 reciprocal scaling, precision={optimized}, no checked/portable promise",
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
        "independent native untimed phase witness/{}:{N}: submission={submission:?}, completion+reduction+release={completion:?}; one call, no confidence interval",
        T::LABEL
    );
    let mut memory = backend.memory_provider(pool);
    let mut observed = vec![T::default(); 1];
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
            "cuda_native_mse_f64/{optimized}/{}/{boundary}/{N}",
            T::LABEL
        ));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(u64::try_from(N)?));
        for route in [
            "actual_source",
            "explicit_graph",
            "independent_native_asum_scal",
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
                            source::loss(w, g, optimized).unwrap()
                        } else {
                            (if optimized {
                                source::optimized::<T, N>(&source_w[bank], &source_g[bank])
                            } else {
                                source::preserve::<T, N>(&source_w[bank], &source_g[bank])
                            })
                            .unwrap()
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
                        let output = native.submit(bank).unwrap();
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
            #[cfg(feature = "allocation-census")]
            {
                fusion_pcu_cuda::reset_cuda_api_census();
                let (_, census) = super::allocations::measure(&mut call);
                let api = fusion_pcu_cuda::cuda_api_census();
                assert_eq!(api.kernel_launches, 1);
                assert_eq!(api.allocations, 2);
                assert_eq!(api.frees, 2);
                assert_eq!(api.module_loads, 0);
                assert_eq!(api.symbol_resolutions, 0);
                eprintln!(
                    "census/{}/{boundary}/{N}/{route}: Rust alloc={} realloc={} frees={} bytes={}; API={api:?}; resident API/Rust also includes untimed readback/oracle",
                    T::LABEL,
                    census.alloc_calls,
                    census.realloc_calls,
                    census.dealloc_calls,
                    census.requested_bytes
                );
            }
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
        }
        group.finish();
    }
    Ok(())
}
pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    super::activity::activity_guard();
    let (_discovery, backend, runtime) = super::selection::selected_device();
    for optimized in [false, true] {
        case::<f64, 65>(criterion, &backend, &runtime, optimized)?;
        case::<f64, 1_048_576>(criterion, &backend, &runtime, optimized)?;
    }
    global::clear_thread_cache()?;
    Ok(())
}

//! API boundary comparison with changing inputs, complete readback, and escaped-output drops.
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
    PcuBindingRef,
    PcuCompoundArithmeticPolicy,
    PcuDeviceTensor,
    PcuHostArgument,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuNumericalOptions,
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
    HipRuntime,
    RocmOwnedDispatchBackend,
    RocmOwnedTensorAssessor,
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

#[allow(clippy::too_many_lines)] // Keep matching source/graph/native allocation and timing boundaries visible.
fn case<T: Scalar, const R: usize, const K: usize, const C: usize>(
    criterion: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    runtime: &HipRuntime,
) -> Result<(), Box<dyn Error>> {
    super::activity::guard();
    let pool = PcuMemoryPoolId(0);
    let assessor_root = RocmOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = assessor_root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        ..Default::default()
    });
    let left_id = graph.input([R, K], T::TYPE)?;
    let right_id = graph.input([K, C], T::TYPE)?;
    let output_id = graph.matmul(left_id, right_id)?;
    let program = graph.into_selected_program(
        &[output_id],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = assessor.prepare_owned_program(program)?;
    let mut memory = backend.memory_provider(pool);
    let mut native = Native::new::<T, R, K, C>(runtime)?;
    let inputs = [oracle::fill::<T, R, K, C>(0), oracle::fill::<T, R, K, C>(1)];
    let source_left = [
        source::identity::<T, R, K>(&inputs[0].0)?,
        source::identity::<T, R, K>(&inputs[1].0)?,
    ];
    let source_right = [
        source::identity::<T, K, C>(&inputs[0].1)?,
        source::identity::<T, K, C>(&inputs[1].1)?,
    ];
    let mut raw_left = [
        PcuDeviceTensor::new(
            [R, K],
            backend.upload_buffer(pool, inputs[0].0.as_flattened())?,
        )?,
        PcuDeviceTensor::new(
            [R, K],
            backend.upload_buffer(pool, inputs[1].0.as_flattened())?,
        )?,
    ];
    let mut raw_right = [
        PcuDeviceTensor::new(
            [K, C],
            backend.upload_buffer(pool, inputs[0].1.as_flattened())?,
        )?,
        PcuDeviceTensor::new(
            [K, C],
            backend.upload_buffer(pool, inputs[1].1.as_flattened())?,
        )?,
    ];
    let mut observed = vec![T::default(); R * C];
    let expected = [
        oracle::expected::<T, R, K, C>(0),
        oracle::expected::<T, R, K, C>(1),
    ];
    // Every route is executed and independently checked before any census or timing.
    for bank in 0..2 {
        native.upload(
            bank,
            inputs[bank].0.as_flattened(),
            inputs[bank].1.as_flattened(),
        )?;
        let output = source::product::<T, R, K, C>(&source_left[bank], &source_right[bank])?;
        output.read_into(&mut observed)?;
        oracle::verify(&expected[bank], &observed);
        let output = assessor.execute_owned_program_outputs(
            &prepared,
            &[(left_id, &raw_left[bank]), (right_id, &raw_right[bank])],
            pool,
            &mut memory,
        )?;
        backend.download_buffer(pool, output[0].1.buffer(), &mut observed)?;
        oracle::verify(&expected[bank], &observed);
        let output = native.submit::<T, R, K, C>(bank)?;
        Native::read(&output, &mut observed)?;
        oracle::verify(&expected[bank], &observed);
    }
    eprintln!(
        "native/{}/{R}x{K}x{C}: bounded exact fixture only; BackendDefined + Preserve + Unspecified; fresh outputs, no fault status; same terminal batch event boundary for all three routes; direct control is the safe typed rocBLAS API including its ownership work; full-host uploads/readback included; resident readback excluded; API boundary comparison, no isolated scheduling claim",
        T::LABEL
    );
    #[cfg(feature = "allocation-census")]
    if cfg!(feature = "allocation-census") {
        for route in [
            "source_native",
            "explicit_graph_native",
            "native_typed_rocblas_fresh",
        ] {
            super::allocations::census(
                &format!("{}/{R}x{K}x{C}/resident/{route}", T::LABEL),
                || match route {
                    "source_native" => drop(
                        source::product::<T, R, K, C>(&source_left[0], &source_right[0]).unwrap(),
                    ),
                    "explicit_graph_native" => drop(
                        assessor
                            .execute_owned_program_outputs(
                                &prepared,
                                &[(left_id, &raw_left[0]), (right_id, &raw_right[0])],
                                pool,
                                &mut memory,
                            )
                            .unwrap(),
                    ),
                    _ => drop(native.submit::<T, R, K, C>(0).unwrap()),
                },
            );
        }
        return Ok(());
    }
    // Precompute changing host data and the independent integer oracle before timing;
    // repeatedly regenerating a K-wide oracle would dominate process wall for custom durations.
    let host_inputs: Vec<_> = (0..oracle::phase_count::<K>())
        .map(oracle::fill::<T, R, K, C>)
        .collect();
    let host_expected: Vec<_> = (0..oracle::phase_count::<K>())
        .map(oracle::expected::<T, R, K, C>)
        .collect();
    for host in [true, false] {
        // Host profiles overwrite bank zero. Reestablish both immutable resident fixtures
        // before the next boundary, so alternating resident inputs cannot inherit host data.
        for bank in 0..2 {
            memory
                .transfer_to(
                    raw_left[bank].resource_mut(),
                    0,
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), inputs[bank].0.as_flattened())
                        .bytes(),
                )
                .expect("restore resident input bank");
            memory
                .transfer_to(
                    raw_right[bank].resource_mut(),
                    0,
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), inputs[bank].1.as_flattened())
                        .bytes(),
                )
                .expect("restore resident input bank");
            native.upload(
                bank,
                inputs[bank].0.as_flattened(),
                inputs[bank].1.as_flattened(),
            )?;
        }
        let mut measure = |route: &str,
                           host: bool,
                           bank: usize,
                           left: &[[T; K]; R],
                           right: &[[T; C]; K],
                           wanted: &[T]| {
            let mut total = Duration::ZERO;
            let start = Instant::now();
            match route {
                "source_native" => {
                    let output = if host {
                        source::product::<T, R, K, C>(left, right).unwrap()
                    } else {
                        source::product::<T, R, K, C>(&source_left[bank], &source_right[bank])
                            .unwrap()
                    };
                    if host {
                        output.read_into(&mut observed).unwrap();
                    }
                    total += start.elapsed();
                    if !host {
                        output.read_into(&mut observed).unwrap();
                    }
                    oracle::verify(wanted, &observed);
                    let drop_start = Instant::now();
                    drop(black_box(output));
                    total += drop_start.elapsed();
                }
                "explicit_graph_native" => {
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
                    let output = assessor
                        .execute_owned_program_outputs(
                            &prepared,
                            &[(left_id, &raw_left[bank]), (right_id, &raw_right[bank])],
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
                    oracle::verify(wanted, &observed);
                    let drop_start = Instant::now();
                    drop(black_box(output));
                    total += drop_start.elapsed();
                }
                _ => {
                    if host {
                        native
                            .upload(0, left.as_flattened(), right.as_flattened())
                            .unwrap();
                    }
                    let output = native.submit::<T, R, K, C>(bank).unwrap();
                    if host {
                        Native::read(&output, &mut observed).unwrap();
                    }
                    total += start.elapsed();
                    if !host {
                        Native::read(&output, &mut observed).unwrap();
                    }
                    oracle::verify(wanted, &observed);
                    let drop_start = Instant::now();
                    drop(black_box(output));
                    total += drop_start.elapsed();
                }
            }
            total
        };
        let boundary = if host {
            "full_host"
        } else {
            "resident_fresh_output"
        };
        if host {
            // Cover every precomputed changing host phase independently of Criterion's sample
            // count; the heavy profile deliberately retains only its three bounded phases.
            for (phase, (left, right)) in host_inputs.iter().enumerate() {
                for route in [
                    "source_native",
                    "explicit_graph_native",
                    "native_typed_rocblas_fresh",
                ] {
                    let _elapsed = measure(route, true, 0, left, right, &host_expected[phase]);
                }
            }
        }
        if !host {
            // Criterion --test may visit only bank one. Check both restored banks through
            // every actual route before measurement, independently of Criterion iteration count.
            for bank in 0..2 {
                for route in [
                    "source_native",
                    "explicit_graph_native",
                    "native_typed_rocblas_fresh",
                ] {
                    let _elapsed = measure(
                        route,
                        false,
                        bank,
                        &inputs[bank].0,
                        &inputs[bank].1,
                        &expected[bank],
                    );
                }
            }
        }
        if std::env::var_os("PCU_NATIVE_PAIRED").is_some() {
            let routes = [
                "source_native",
                "explicit_graph_native",
                "native_typed_rocblas_fresh",
            ];
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
                let phase = sample % host_inputs.len();
                let bank = if host { 0 } else { sample & 1 };
                let (left, right) = &host_inputs[phase];
                let wanted = if host {
                    &host_expected[phase]
                } else {
                    &expected[bank]
                };
                let mut times = [Duration::ZERO; 3];
                for route in orders[sample % orders.len()] {
                    times[route] = measure(routes[route], host, bank, left, right, wanted);
                }
                for route in 0..2 {
                    ratios[route][sample] = times[route].as_secs_f64() / times[2].as_secs_f64();
                }
            }
            for (label, mut samples) in ["source/native", "graph/native"].into_iter().zip(ratios) {
                samples.sort_by(f64::total_cmp);
                let median = samples[17].midpoint(samples[18]);
                eprintln!(
                    "paired/{}/{boundary}/{R}x{K}x{C}/{label}: median={median:.6}; 36 triples, six balanced orders; observed ratios without confidence interval",
                    T::LABEL
                );
            }
            continue;
        }
        let mut group = criterion.benchmark_group(format!(
            "rocm_native_matmul/{}/{boundary}/{R}x{K}x{C}",
            T::LABEL
        ));
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(u64::try_from(R * C)?));
        for route in [
            "source_native",
            "explicit_graph_native",
            "native_typed_rocblas_fresh",
        ] {
            let mut job = 0_u64;
            group.bench_function(route, |bench| {
                bench.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iterations {
                        job = job.wrapping_add(1);
                        let phase = usize::try_from(job % oracle::phase_count::<K>()).unwrap();
                        let bank = if host {
                            0
                        } else {
                            usize::try_from(job & 1).unwrap()
                        };
                        let (left, right) = &host_inputs[phase];
                        let wanted = if host {
                            &host_expected[phase]
                        } else {
                            &expected[bank]
                        };
                        total += measure(route, host, bank, left, right, wanted);
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
    super::activity::guard();
    let (_discovery, backend, runtime) = super::selection::selected_device();
    case::<f32, 4, 4, 4>(criterion, &backend, &runtime)?;
    case::<f64, 4, 4, 4>(criterion, &backend, &runtime)?;
    case::<f32, 32, 32, 32>(criterion, &backend, &runtime)?;
    case::<f64, 32, 32, 32>(criterion, &backend, &runtime)?;
    case::<f32, 64, 256, 64>(criterion, &backend, &runtime)?;
    case::<f64, 64, 256, 64>(criterion, &backend, &runtime)?;
    case::<f32, 1024, 2048, 1024>(criterion, &backend, &runtime)?;
    case::<f64, 1024, 2048, 1024>(criterion, &backend, &runtime)?;
    global::clear_thread_cache()?;
    Ok(())
}

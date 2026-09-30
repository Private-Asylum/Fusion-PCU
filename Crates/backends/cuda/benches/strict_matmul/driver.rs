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

#[allow(clippy::too_many_lines)] // Keep exact source/graph/native timing boundaries reviewable together.
fn case<T: Scalar, const R: usize, const K: usize, const C: usize>(
    criterion: &mut Criterion,
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
) -> Result<(), Box<dyn Error>> {
    super::activity::guard();
    let pool = PcuMemoryPoolId(0);
    let assessor_root = CudaOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = assessor_root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left_id = graph.input([R, K], T::TYPE)?;
    let right_id = graph.input([K, C], T::TYPE)?;
    let output_id = graph.matmul(left_id, right_id)?;
    let mut native = Native::new::<T, R, K, C>(runtime, &graph, output_id)?;
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
    let Err(boundary_error) = source::boundary::<T, R, K, C>(&left, &right) else {
        panic!("boundary mode must remain unadmitted");
    };
    assert!(
        super::correctness::boundary_rejected(&boundary_error),
        "unexpected boundary admission error: {boundary_error:?}"
    );
    eprintln!(
        "{} {R}x{K}x{C}: Boundary Unsupported (admission assertion, not compute timing); strict ordered separate multiply/add; block=256; source/graph/native fresh outputs and private fault status; vendor preallocated output; vendor BLAS different unchecked semantics",
        T::LABEL
    );
    #[cfg(feature = "allocation-census")]
    if cfg!(feature = "allocation-census") {
        let source_left = source::identity::<T, R, K>(&left)?;
        let source_right = source::identity::<T, K, C>(&right)?;
        let raw_left =
            PcuDeviceTensor::new([R, K], backend.upload_buffer(pool, left.as_flattened())?)?;
        let raw_right =
            PcuDeviceTensor::new([K, C], backend.upload_buffer(pool, right.as_flattened())?)?;
        native.upload(left.as_flattened(), right.as_flattened())?;
        // Warm every route before capture; output ownership/free is included, readback is excluded.
        drop(source::product::<T, R, K, C>(&source_left, &source_right)?);
        drop(assessor.execute_owned_program_outputs(
            &prepared,
            &[(left_id, &raw_left), (right_id, &raw_right)],
            pool,
            &mut memory,
        )?);
        drop(native.submit_fresh(0)?);
        native.vendor::<T, R, K, C>(0)?;
        for route in [
            "source_strict",
            "explicit_graph_strict",
            "native_same_ordered_checker_fresh",
            "vendor_blas_unchecked_different_semantics",
        ] {
            super::allocations::census(
                &format!("{}/{R}x{K}x{C}/resident/{route}", T::LABEL),
                || match route {
                    "source_strict" => {
                        drop(source::product::<T, R, K, C>(&source_left, &source_right).unwrap());
                    }
                    "explicit_graph_strict" => drop(
                        assessor
                            .execute_owned_program_outputs(
                                &prepared,
                                &[(left_id, &raw_left), (right_id, &raw_right)],
                                pool,
                                &mut memory,
                            )
                            .unwrap(),
                    ),
                    "native_same_ordered_checker_fresh" => {
                        let (output, fault) = native.submit_fresh(0).unwrap();
                        assert_eq!(fault, u64::MAX);
                        drop(output);
                    }
                    _ => native.vendor::<T, R, K, C>(0).unwrap(),
                },
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
        if !host {
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
        let mut group = criterion.benchmark_group(format!(
            "cuda_strict_matmul/{}/{boundary}/{R}x{K}x{C}",
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
        group.finish();
    }
    Ok(())
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    super::activity::guard();
    let (_discovery, backend, runtime) = super::selection::selected_device();
    super::correctness::run::<f32>(&backend, &runtime)?;
    super::correctness::run::<f64>(&backend, &runtime)?;
    case::<f32, 4, 2, 4>(criterion, &backend, &runtime)?;
    case::<f64, 4, 2, 4>(criterion, &backend, &runtime)?;
    case::<f32, 32, 32, 32>(criterion, &backend, &runtime)?;
    case::<f64, 32, 32, 32>(criterion, &backend, &runtime)?;
    case::<f32, 64, 256, 64>(criterion, &backend, &runtime)?;
    case::<f64, 64, 256, 64>(criterion, &backend, &runtime)?;
    global::clear_thread_cache()?;
    Ok(())
}

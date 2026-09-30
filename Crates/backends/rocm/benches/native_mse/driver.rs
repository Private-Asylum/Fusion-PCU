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
    PcuScalarType,
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
    oracle,
    source,
};

#[allow(clippy::too_many_lines)] // Keep matching source/graph/native allocation and timing boundaries visible.
fn case<const N: usize>(
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
    let left_id = graph.input([N], PcuScalarType::F32)?;
    let right_id = graph.input([N], PcuScalarType::F32)?;
    let output_id = graph.mean_squared_error(left_id, right_id)?;
    let program = graph.into_selected_program(
        &[output_id],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = assessor.prepare_owned_program(program)?;
    let mut memory = backend.memory_provider(pool);
    let mut native = Native::new::<N>(runtime)?;
    let inputs = [oracle::inputs::<N>(0), oracle::inputs::<N>(1)];
    let source_left = [
        source::identity::<N>(&inputs[0].0)?,
        source::identity::<N>(&inputs[1].0)?,
    ];
    let source_right = [
        source::identity::<N>(&inputs[0].1)?,
        source::identity::<N>(&inputs[1].1)?,
    ];
    let mut raw_left = [
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[0].0.as_slice())?)?,
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[1].0.as_slice())?)?,
    ];
    let mut raw_right = [
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[0].1.as_slice())?)?,
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[1].1.as_slice())?)?,
    ];
    let mut observed = vec![0.0_f32; 1];
    let expected = [oracle::expected::<N>(0), oracle::expected::<N>(1)];
    // Every route is executed and independently checked before any census or timing.
    for bank in 0..2 {
        native.upload(bank, inputs[bank].0.as_slice(), inputs[bank].1.as_slice())?;
        let output = source::loss::<N>(&source_left[bank], &source_right[bank])?;
        output.read_into(&mut observed)?;
        oracle::verify(expected[bank], &observed);
        let output = assessor.execute_owned_program_outputs(
            &prepared,
            &[(left_id, &raw_left[bank]), (right_id, &raw_right[bank])],
            pool,
            &mut memory,
        )?;
        backend.download_buffer(pool, output[0].1.buffer(), &mut observed)?;
        oracle::verify(expected[bank], &observed);
        let output = native.submit(bank)?;
        Native::read(&output, &mut observed)?;
        oracle::verify(expected[bank], &observed);
    }
    eprintln!(
        "native/f32/{N}: exact squared-integer sum then rounded F32 reciprocal scale; BackendDefined + Preserve + Unspecified; fresh squared scratch and scalar output, no fault status; square kernel event wait then synchronous typed ASUM/scaling boundary for all three routes; full-host uploads/readback included; resident readback excluded; API boundary comparison, no isolated scheduling claim"
    );
    #[cfg(feature = "allocation-census")]
    if cfg!(feature = "allocation-census") {
        for route in [
            "source_native_mse",
            "explicit_graph_native_mse",
            "native_typed_asum_fresh",
        ] {
            super::allocations::census(
                &format!("{}/{N}/resident/{route}", "f32"),
                || match route {
                    "source_native_mse" => {
                        drop(source::loss::<N>(&source_left[0], &source_right[0]).unwrap());
                    }
                    "explicit_graph_native_mse" => drop(
                        assessor
                            .execute_owned_program_outputs(
                                &prepared,
                                &[(left_id, &raw_left[0]), (right_id, &raw_right[0])],
                                pool,
                                &mut memory,
                            )
                            .unwrap(),
                    ),
                    _ => drop(native.submit(0).unwrap()),
                },
            );
        }
        return Ok(());
    }
    // Precompute changing host data and the independent integer oracle before timing;
    // deriving a million-element oracle inside the loop would obscure API measurements.
    let host_inputs: Vec<_> = (0..3_u32).map(oracle::inputs::<N>).collect();
    let host_expected: Vec<_> = (0..3_u32)
        .map(|phase| [oracle::expected::<N>(phase)])
        .collect();
    for host in [true, false] {
        // Host profiles overwrite bank zero. Reestablish both immutable resident fixtures
        // before the next boundary, so alternating resident inputs cannot inherit host data.
        for bank in 0..2 {
            memory
                .transfer_to(
                    raw_left[bank].resource_mut(),
                    0,
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), inputs[bank].0.as_slice())
                        .bytes(),
                )
                .expect("restore resident input bank");
            memory
                .transfer_to(
                    raw_right[bank].resource_mut(),
                    0,
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), inputs[bank].1.as_slice())
                        .bytes(),
                )
                .expect("restore resident input bank");
            native.upload(bank, inputs[bank].0.as_slice(), inputs[bank].1.as_slice())?;
        }
        let mut measure = |route: &str,
                           host: bool,
                           bank: usize,
                           left: &[f32; N],
                           right: &[f32; N],
                           wanted: &[f32]| {
            let mut total = Duration::ZERO;
            let start = Instant::now();
            match route {
                "source_native_mse" => {
                    let output = if host {
                        source::loss::<N>(left, right).unwrap()
                    } else {
                        source::loss::<N>(&source_left[bank], &source_right[bank]).unwrap()
                    };
                    if host {
                        output.read_into(&mut observed).unwrap();
                    }
                    total += start.elapsed();
                    if !host {
                        output.read_into(&mut observed).unwrap();
                    }
                    oracle::verify(wanted[0], &observed);
                    let drop_start = Instant::now();
                    drop(black_box(output));
                    total += drop_start.elapsed();
                }
                "explicit_graph_native_mse" => {
                    if host {
                        memory
                            .transfer_to(
                                raw_left[0].resource_mut(),
                                0,
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), left.as_slice())
                                    .bytes(),
                            )
                            .unwrap();
                        memory
                            .transfer_to(
                                raw_right[0].resource_mut(),
                                0,
                                PcuHostArgument::read(PcuBindingRef::new(0, 1), right.as_slice())
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
                    oracle::verify(wanted[0], &observed);
                    let drop_start = Instant::now();
                    drop(black_box(output));
                    total += drop_start.elapsed();
                }
                _ => {
                    if host {
                        native.upload(0, left.as_slice(), right.as_slice()).unwrap();
                    }
                    let output = native.submit(bank).unwrap();
                    if host {
                        Native::read(&output, &mut observed).unwrap();
                    }
                    total += start.elapsed();
                    if !host {
                        Native::read(&output, &mut observed).unwrap();
                    }
                    oracle::verify(wanted[0], &observed);
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
            // count; both profiles retain three independently checked phases.
            for (phase, (left, right)) in host_inputs.iter().enumerate() {
                for route in [
                    "source_native_mse",
                    "explicit_graph_native_mse",
                    "native_typed_asum_fresh",
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
                    "source_native_mse",
                    "explicit_graph_native_mse",
                    "native_typed_asum_fresh",
                ] {
                    let _elapsed = measure(
                        route,
                        false,
                        bank,
                        &inputs[bank].0,
                        &inputs[bank].1,
                        &[expected[bank]],
                    );
                }
            }
        }
        if std::env::var_os("PCU_MSE_PAIRED").is_some() {
            let routes = [
                "source_native_mse",
                "explicit_graph_native_mse",
                "native_typed_asum_fresh",
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
                    &[expected[bank]]
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
                    "paired/f32/{boundary}/{N}/{label}: median={median:.6}; 36 triples, six balanced orders; observed ratios without confidence interval"
                );
            }
            continue;
        }
        let mut group = criterion.benchmark_group(format!("rocm_native_mse/f32/{boundary}/{N}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(u64::try_from(N)?));
        for route in [
            "source_native_mse",
            "explicit_graph_native_mse",
            "native_typed_asum_fresh",
        ] {
            let mut job = 0_u64;
            group.bench_function(route, |bench| {
                bench.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iterations {
                        job = job.wrapping_add(1);
                        let phase = usize::try_from(job % 3).unwrap();
                        let bank = if host {
                            0
                        } else {
                            usize::try_from(job & 1).unwrap()
                        };
                        let (left, right) = &host_inputs[phase];
                        let wanted = if host {
                            &host_expected[phase]
                        } else {
                            &[expected[bank]]
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
    case::<65>(criterion, &backend, &runtime)?;
    case::<1_048_576>(criterion, &backend, &runtime)?;
    global::clear_thread_cache()?;
    Ok(())
}

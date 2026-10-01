//! Synchronous API wall time; full readback and numerical checks surround each timed sample.
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
    PcuHostArgument,
    PcuMemoryProvider,
    PcuCompoundArithmeticPolicy,
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
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

const ROUTES: [&str; 3] = [
    "source_native_sgd",
    "explicit_graph_native_sgd",
    "handwritten_hip_fresh_output",
];

#[allow(clippy::too_many_lines)] // Keep matched ownership, validation and timing boundaries visible together.
fn case<const N: usize>(
    criterion: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    runtime: &HipRuntime,
) -> Result<(), Box<dyn Error>> {
    let pool = PcuMemoryPoolId(0);
    let cold = Instant::now();
    let assessor_root = RocmOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = assessor_root.assessor();
    let mut memory = backend.memory_provider(pool);
    let mut native = Native::new::<N>(runtime)?;
    let inputs = [oracle::inputs::<N>(0), oracle::inputs::<N>(1)];
    let left = [
        source::identity::<N>(&inputs[0].0)?,
        source::identity::<N>(&inputs[1].0)?,
    ];
    let right = [
        source::identity::<N>(&inputs[0].1)?,
        source::identity::<N>(&inputs[1].1)?,
    ];
    let raw_left = [
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[0].0.as_slice())?)?,
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[1].0.as_slice())?)?,
    ];
    let raw_right = [
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[0].1.as_slice())?)?,
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[1].1.as_slice())?)?,
    ];
    let mut staged_left =
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[0].0.as_slice())?)?;
    let mut staged_right =
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, inputs[0].1.as_slice())?)?;
    for (bank, (weights, gradient)) in inputs.iter().enumerate() {
        native.upload(bank, weights.as_slice(), gradient.as_slice())?;
    }
    eprintln!(
        "cold/f32/{N}/fixture_and_handwritten_compile: {:?}; source SGD lazy preparation included in its first preflight visit when needed",
        cold.elapsed()
    );
    for contracted in [false, true] {
        let mode = if contracted {
            "backend_precision_fma"
        } else {
            "preserve_mul_sub"
        };
        let mut graph = Graph::default();
        graph.set_numerical_options(PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: if contracted {
                PcuPrecisionPolicy::BackendOptimized
            } else {
                PcuPrecisionPolicy::Preserve
            },
            ..Default::default()
        });
        let weights_id = graph.input([N], PcuScalarType::F32)?;
        let gradient_id = graph.input([N], PcuScalarType::F32)?;
        let output_id = graph.sgd_update(weights_id, gradient_id, oracle::RATE)?;
        let program = graph.into_selected_program(
            &[output_id],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )?;
        let cold = Instant::now();
        let prepared = assessor.prepare_owned_program(program)?;
        eprintln!("cold/f32/{N}/{mode}/graph_prepare: {:?}", cold.elapsed());
        let expected: Vec<_> = inputs
            .iter()
            .map(|(weights, gradient)| {
                oracle::expected(weights.as_slice(), gradient.as_slice(), contracted)
            })
            .collect();
        assert_eq!(
            expected[0][0].to_bits(),
            if contracted { 0x2880_0000 } else { 0 }
        );
        let mut observed = vec![0.0; N];
        let mut measure = |route: &str, host: bool, bank: usize| {
            let (weights, gradient) = &inputs[bank];
            let start = Instant::now();
            let mut elapsed;
            match route {
                "source_native_sgd" => {
                    let output = if host {
                        if contracted {
                            source::contracted::<N>(weights, gradient)
                        } else {
                            source::preserved::<N>(weights, gradient)
                        }
                    } else if contracted {
                        source::contracted::<N>(&left[bank], &right[bank])
                    } else {
                        source::preserved::<N>(&left[bank], &right[bank])
                    }
                    .unwrap();
                    if host {
                        output.read_into(&mut observed).unwrap();
                    }
                    elapsed = start.elapsed();
                    if !host {
                        output.read_into(&mut observed).unwrap();
                    }
                    oracle::verify(&expected[bank], &observed);
                    let drop_start = Instant::now();
                    drop(black_box(output));
                    elapsed += drop_start.elapsed();
                }
                "explicit_graph_native_sgd" => {
                    if host {
                        memory
                            .transfer_to(
                                staged_left.resource_mut(),
                                0,
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), weights.as_slice())
                                    .bytes(),
                            )
                            .unwrap();
                        memory
                            .transfer_to(
                                staged_right.resource_mut(),
                                0,
                                PcuHostArgument::read(
                                    PcuBindingRef::new(0, 1),
                                    gradient.as_slice(),
                                )
                                .bytes(),
                            )
                            .unwrap();
                    }
                    let (weights, gradient) = if host {
                        (&staged_left, &staged_right)
                    } else {
                        (&raw_left[bank], &raw_right[bank])
                    };
                    let output = assessor
                        .execute_owned_program_outputs(
                            &prepared,
                            &[(weights_id, weights), (gradient_id, gradient)],
                            pool,
                            &mut memory,
                        )
                        .unwrap();
                    if host {
                        backend
                            .download_buffer(pool, output[0].1.buffer(), &mut observed)
                            .unwrap();
                    }
                    elapsed = start.elapsed();
                    if !host {
                        backend
                            .download_buffer(pool, output[0].1.buffer(), &mut observed)
                            .unwrap();
                    }
                    oracle::verify(&expected[bank], &observed);
                    let drop_start = Instant::now();
                    drop(black_box(output));
                    elapsed += drop_start.elapsed();
                }
                _ => {
                    let output = if host {
                        native.host(weights.as_slice(), gradient.as_slice(), contracted)
                    } else {
                        native.submit(bank, contracted)
                    }
                    .unwrap();
                    if host {
                        Native::read(&output, &mut observed).unwrap();
                    }
                    elapsed = start.elapsed();
                    if !host {
                        Native::read(&output, &mut observed).unwrap();
                    }
                    oracle::verify(&expected[bank], &observed);
                    let drop_start = Instant::now();
                    drop(black_box(output));
                    elapsed += drop_start.elapsed();
                }
            }
            elapsed
        };
        for host in [true, false] {
            let boundary = if host {
                "full_host_reused_upload_storage"
            } else {
                "resident_launch_wait_fresh_output"
            };
            for route in ROUTES {
                for bank in 0..2 {
                    let first = measure(route, host, bank);
                    eprintln!(
                        "preflight/f32/{N}/{mode}/{boundary}/{route}/bank{bank}: {first:?}; full output oracle passed (source cache misses may include lazy preparation; preflight samples are not performance statistics)"
                    );
                }
            }
            eprintln!(
                "f32/{N}/{mode}/{boundary}: immutable input banks alternate; fresh output retained through full readback then dropped; launch and terminal completion wait included; resident readback/oracle excluded from time; full-host upload/readback included with reused input storage on every route; no isolated submission, completion, GPU-event, device-allocation or API-call census measurements"
            );
            #[cfg(feature = "allocation-census")]
            if cfg!(feature = "allocation-census") {
                for route in ROUTES {
                    // Census encloses the real checked profile: includes readback/oracle and
                    // output drop even where iter_custom excludes readback from wall time.
                    super::allocations::census(
                        &format!("f32/{N}/{mode}/{boundary}/{route}/checked_profile"),
                        || {
                            black_box(measure(route, host, 0));
                        },
                    );
                }
                continue;
            }
            {
                let mut group =
                    criterion.benchmark_group(format!("rocm_native_sgd/f32/{N}/{mode}/{boundary}"));
                group.sample_size(20);
                group.warm_up_time(Duration::from_secs(1));
                group.measurement_time(Duration::from_secs(2));
                group.throughput(Throughput::Elements(u64::try_from(N)?));
                for route in ROUTES {
                    let mut bank = 0;
                    group.bench_function(route, |bench| {
                        bench.iter_custom(|iterations| {
                            let mut total = Duration::ZERO;
                            for _ in 0..iterations {
                                bank ^= 1;
                                total += measure(route, host, bank);
                            }
                            total
                        });
                    });
                }
                group.finish();
            }
        }
    }
    Ok(())
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    super::activity::guard();
    let (_discovery, backend, runtime) = super::selection::selected_device();
    case::<17>(criterion, &backend, &runtime)?;
    case::<4096>(criterion, &backend, &runtime)?;
    case::<1_048_576>(criterion, &backend, &runtime)?;
    global::clear_thread_cache()?;
    Ok(())
}

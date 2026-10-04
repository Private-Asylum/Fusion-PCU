//! Two real resident training steps plus final readback, preserving every route's storage cost.
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCompoundArithmeticPolicy,
    PcuDeviceTensor,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuScalarType,
    PcuTensor,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    Tensor,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
    TensorValue,
    ValueId,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
};
#[rustfmt::skip]
use std::{
    error::Error,
    sync::atomic::{
        AtomicUsize,
        Ordering,
    },
};
#[rustfmt::skip]
use super::{
    native,
    source,
    support,
    train_step_reference::verify,
};

static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

#[allow(clippy::unnecessary_box_returns)] // Cold fixed matrices stay on the heap to bound stack use.
fn matrix<const R: usize, const K: usize>(data: &[f32]) -> Box<[[f32; K]; R]> {
    assert_eq!(data.len(), R * K);
    let (rows, tail) = data.as_chunks::<K>();
    assert!(tail.is_empty());
    let rows = rows.to_vec();
    rows.into_boxed_slice().try_into().unwrap()
}

struct Sources {
    samples: PcuTensor<f32>,
    weights: [PcuTensor<f32>; 2],
    target: PcuTensor<f32>,
    rate: PcuTensor<f32>,
}
impl Sources {
    fn two<const R: usize, const K: usize>(
        &self,
        kind: u8,
        bank: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let call = |weights: &PcuTensor<f32>| match kind {
            0 => {
                source::multiply_subtract::<R, K>(&self.samples, weights, &self.target, &self.rate)
            }
            1 => source::separate_update::<R, K>(&self.samples, weights, &self.target),
            _ => source::checked::<R, K>(&self.samples, weights, &self.target),
        };
        let first = call(&self.weights[bank])?;
        let second = call(&first)?;
        let mut output = vec![0.0; K];
        second.read_into(&mut output)?;
        Ok(output)
    }
}

struct Program {
    graph: Graph,
    inputs: Vec<ValueId>,
    output: ValueId,
}
fn program(rows: usize, features: usize, kind: u8) -> Result<Program, Box<dyn Error>> {
    let mut graph = Graph::default();
    if kind == 2 {
        graph.set_numerical_mode(PcuNumericalMode::Strict);
    } else {
        graph.set_numerical_options(PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            ..PcuNumericalOptions::default()
        });
    }
    let samples = graph.input([rows, features], PcuScalarType::F32)?;
    let weights = graph.input([features, 1], PcuScalarType::F32)?;
    let target = graph.input([rows, 1], PcuScalarType::F32)?;
    let rate = if kind == 0 {
        Some(graph.input([features, 1], PcuScalarType::F32)?)
    } else {
        None
    };
    let prediction = graph.matmul(samples, weights)?;
    let loss = graph.mean_squared_error(prediction, target)?;
    let gradient = graph.backward_mse_for(loss, weights)?;
    let output = if kind == 0 {
        let scaled = graph.mul(gradient, rate.unwrap())?;
        graph.sub(weights, scaled)?
    } else {
        graph.sgd_update(weights, gradient, 0.001)?
    };
    let mut inputs = vec![samples, weights, target];
    if let Some(rate) = rate {
        inputs.push(rate);
    }
    Ok(Program {
        graph,
        inputs,
        output,
    })
}

fn expected(
    program: &Program,
    kind: u8,
    samples: &Tensor,
    weights: &Tensor,
    target: &Tensor,
    rate: &Tensor,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut weights = weights.clone();
    for _ in 0..2 {
        let inputs = program
            .inputs
            .iter()
            .copied()
            .zip([
                samples.clone(),
                weights.clone(),
                target.clone(),
                rate.clone(),
            ])
            .map(|(id, value)| (id, TensorValue::F32(value)))
            .collect::<Vec<_>>();
        let execution = if kind == 2 {
            program.graph.evaluate_checked(&inputs)?
        } else {
            program.graph.evaluate(&inputs)?
        };
        weights = execution.value_typed::<f32>(program.output)?.clone();
    }
    Ok(weights.into_data())
}

#[allow(clippy::too_many_lines)] // Freeze the exact source, graph, library and output ownership boundaries together.
pub fn case<const R: usize, const K: usize>(
    criterion: &mut Criterion,
    discovery: &RocmDiscovery,
    backend: &RocmOwnedDispatchBackend,
    selected: &support::selection::Candidate,
    inputs: [&[f32]; 4],
) -> Result<(), Box<dyn Error>> {
    let [samples, target, initial, rate] = inputs;
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(selected.device.id),
        block_size: 64,
        score_invocation: Some(score),
        ..Default::default()
    })?;
    let mut changed = initial.to_vec();
    changed[0] += 0.125;
    let weights = [initial.to_vec(), changed];
    let host_samples = Tensor::new([R, K], samples.to_vec())?;
    let host_target = Tensor::new([R, 1], target.to_vec())?;
    let host_rate = Tensor::new([K, 1], rate.to_vec())?;
    let host_weights = weights
        .each_ref()
        .map(|w| Tensor::new([K, 1], w.clone()).unwrap());
    let sources = Sources {
        samples: source::retain(matrix::<R, K>(samples).as_ref())?,
        weights: weights
            .each_ref()
            .map(|w| source::retain(matrix::<K, 1>(w).as_ref()).unwrap()),
        target: source::retain(matrix::<R, 1>(target).as_ref())?,
        rate: source::retain(matrix::<K, 1>(rate).as_ref())?,
    };
    let assessor = RocmTensorAssessor::new(backend)?;
    let mut memory = backend.memory_provider(selected.pool);
    let device_samples =
        PcuDeviceTensor::new([R, K], backend.upload_buffer(selected.pool, samples)?)?;
    let device_target =
        PcuDeviceTensor::new([R, 1], backend.upload_buffer(selected.pool, target)?)?;
    let device_rate = PcuDeviceTensor::new([K, 1], backend.upload_buffer(selected.pool, rate)?)?;
    let device_weights = weights.each_ref().map(|w| {
        PcuDeviceTensor::new([K, 1], backend.upload_buffer(selected.pool, w).unwrap()).unwrap()
    });
    let samples_ref = assessor.borrow_device_input_ref(&device_samples, selected.pool)?;
    let target_ref = assessor.borrow_device_input_ref(&device_target, selected.pool)?;
    let rate_ref = assessor.borrow_device_input_ref(&device_rate, selected.pool)?;
    let weights_ref = device_weights
        .each_ref()
        .map(|w| assessor.borrow_device_input_ref(w, selected.pool).unwrap());
    let factor = vec![2.0 / f32::from(u16::try_from(R)?); R];
    for kind in 0..3 {
        if kind == 2 && R > 4 {
            continue;
        } // Real ordered training has its own bounded family.
        let program = program(R, K, kind)?;
        let expected = host_weights
            .each_ref()
            .map(|w| expected(&program, kind, &host_samples, w, &host_target, &host_rate).unwrap());
        let prepared = assessor.prepare_owned_program(program.graph.into_selected_program(
            &[program.output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )?)?;
        let mut graph = |bank: usize| -> Result<Vec<f32>, Box<dyn Error>> {
            let first_inputs = program
                .inputs
                .iter()
                .copied()
                .zip([&samples_ref, &weights_ref[bank], &target_ref, &rate_ref])
                .collect::<Vec<_>>();
            let first = assessor.execute_owned_program_output_from_inputs::<f32, _>(
                &prepared,
                &first_inputs,
                selected.pool,
                &mut memory,
            )?;
            let first_ref = assessor.borrow_device_input_ref(&first, selected.pool)?;
            let second_inputs = program
                .inputs
                .iter()
                .copied()
                .zip([&samples_ref, &first_ref, &target_ref, &rate_ref])
                .collect::<Vec<_>>();
            let second = assessor.execute_owned_program_output_from_inputs::<f32, _>(
                &prepared,
                &second_inputs,
                selected.pool,
                &mut memory,
            )?;
            let mut output = vec![0.0_f32; K];
            backend.download_buffer(selected.pool, second.buffer(), &mut output)?;
            Ok(output)
        };
        let libraries = if kind == 2 {
            None
        } else {
            Some(weights.each_ref().map(|initial_weights| {
                let inputs = native::TrainInputs {
                    rows: R,
                    features: K,
                    samples,
                    target,
                    factor: &factor,
                    initial_weights,
                    rate,
                };
                native::NativeTrainStep::prepare_with_forward_loss(
                    discovery,
                    selected.device,
                    &inputs,
                    (kind != 0).then_some(0.001),
                )
                .unwrap()
            }))
        };
        for bank in 0..2 {
            let authored = sources.two::<R, K>(kind, bank)?;
            let explicit = graph(bank)?;
            if kind == 2 {
                assert_eq!(
                    authored.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected[bank]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    explicit.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected[bank]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            } else {
                verify(&expected[bank], &authored)?;
                verify(&expected[bank], &explicit)?;
            }
            if let Some(libraries) = &libraries {
                verify(&expected[bank], &libraries[bank].execute_two()?)?;
            }
        }
        #[cfg(feature = "allocation-census")]
        for route in 0..if libraries.is_some() { 3 } else { 2 } {
            let before = SCORES.load(Ordering::Relaxed);
            fusion_pcu_rocm::reset_rocm_api_census();
            let capture = super::alloc::AllocationCapture::start();
            let baseline = super::alloc::AllocationCapture::snapshot();
            for iteration in 0..64 {
                let bank = iteration % 2;
                let actual = match route {
                    0 => sources.two::<R, K>(kind, bank)?,
                    1 => graph(bank)?,
                    _ => libraries.as_ref().unwrap()[bank].execute_two()?,
                };
                verify(&expected[bank], &actual)?;
            }
            let rust = super::alloc::AllocationCapture::finish().since(baseline);
            drop(capture);
            let api = fusion_pcu_rocm::rocm_api_census();
            assert_eq!(api.symbol_resolutions, 0);
            assert_eq!(api.module_loads, 0);
            assert_eq!(SCORES.load(Ordering::Relaxed), before);
            eprintln!(
                "census/source_training/{R}x{K}/{kind}/{route}/64-changing-calls: Rust alloc={} realloc={} frees={} bytes={}; API={api:?}; scores={before}",
                rust.alloc_calls, rust.realloc_calls, rust.dealloc_calls, rust.requested_bytes
            );
        }
        let before = SCORES.load(Ordering::Relaxed);
        let mut phase = 0;
        let mut group = criterion.benchmark_group(format!("rocm_source_training_two_steps/{kind}"));
        group.throughput(Throughput::Elements(u64::try_from(R * K * 2)?));
        group.bench_function(format!("ordinary_fresh_outputs/{R}x{K}"), |bench| {
            bench.iter(|| {
                phase ^= 1;
                std::hint::black_box(sources.two::<R, K>(kind, phase).unwrap());
            });
        });
        group.bench_function(
            format!("explicit_owned_graph_fresh_outputs/{R}x{K}"),
            |bench| {
                bench.iter(|| {
                    phase ^= 1;
                    std::hint::black_box(graph(phase).unwrap());
                });
            },
        );
        if let Some(libraries) = &libraries {
            group.bench_function(
                format!("unchecked_native_full_loss_retained_outputs/{R}x{K}"),
                |bench| {
                    bench.iter(|| {
                        phase ^= 1;
                        std::hint::black_box(libraries[phase].execute_two().unwrap());
                    });
                },
            );
        }
        assert_eq!(SCORES.load(Ordering::Relaxed), before);
        group.finish();
    }
    Ok(())
}

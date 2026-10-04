//! Two resident steps retain all weights and losses; each route publishes the same final values.
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
    activity,
    native,
    source,
    support,
    values,
    verify_weights,
    RATE,
};

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

#[allow(clippy::unnecessary_box_returns)] // Cold multi-million-element matrices stay on the heap.
fn matrix<const R: usize, const K: usize>(data: &[f32]) -> Box<[[f32; K]; R]> {
    assert_eq!(data.len(), R * K);
    let (rows, tail) = data.as_chunks::<K>();
    assert!(tail.is_empty());
    rows.to_vec().into_boxed_slice().try_into().unwrap()
}

struct Sources {
    samples: PcuTensor<f32>,
    targets: PcuTensor<f32>,
    weights: [[PcuTensor<f32>; 3]; 2],
}

impl Sources {
    fn two<const B: usize, const I: usize, const H: usize, const O: usize>(
        &self,
        bank: usize,
        checked: bool,
    ) -> Result<native::MlpWeights, Box<dyn Error>> {
        let initial = &self.weights[bank];
        let call = |w1: &PcuTensor<f32>, w2: &PcuTensor<f32>, w3: &PcuTensor<f32>| {
            if checked {
                source::checked::<B, I, H, O>(&self.samples, w1, w2, w3, &self.targets)
            } else {
                source::train::<B, I, H, O>(&self.samples, w1, w2, w3, &self.targets)
            }
        };
        let first = call(&initial[0], &initial[1], &initial[2])?;
        let second = call(&first.0, &first.1, &first.2)?;
        let read = |owner: &PcuTensor<f32>, elements: usize| {
            let mut output = vec![0.0; elements];
            owner.read_into(&mut output)?;
            Ok::<_, Box<dyn Error>>(output)
        };
        let mut first_loss = [0.0];
        let mut second_loss = [0.0];
        first.3.read_into(&mut first_loss)?;
        second.3.read_into(&mut second_loss)?;
        Ok(native::MlpWeights {
            w1: read(&second.0, I * H)?,
            w2: read(&second.1, H * H)?,
            w3: read(&second.2, H * O)?,
            losses: [first_loss[0], second_loss[0]],
        })
    }
}

struct Program {
    graph: Graph,
    inputs: [ValueId; 5],
    outputs: [ValueId; 4],
}

fn program(
    batch: usize,
    inputs: usize,
    hidden: usize,
    outputs: usize,
) -> Result<Program, Box<dyn Error>> {
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        ..PcuNumericalOptions::default()
    });
    let samples = graph.input([batch, inputs], PcuScalarType::F32)?;
    let w1 = graph.input([inputs, hidden], PcuScalarType::F32)?;
    let w2 = graph.input([hidden, hidden], PcuScalarType::F32)?;
    let w3 = graph.input([hidden, outputs], PcuScalarType::F32)?;
    let targets = graph.input([batch, outputs], PcuScalarType::F32)?;
    let z1 = graph.matmul(samples, w1)?;
    let h1 = graph.relu(z1)?;
    let z2 = graph.matmul(h1, w2)?;
    let h2 = graph.relu(z2)?;
    let prediction = graph.matmul(h2, w3)?;
    let loss = graph.mean_squared_error(prediction, targets)?;
    let gradients = graph.backward_mse_for_targets(loss, &[w1, w2, w3])?;
    let updated = [
        graph.sgd_update(w1, gradients[0], RATE)?,
        graph.sgd_update(w2, gradients[1], RATE)?,
        graph.sgd_update(w3, gradients[2], RATE)?,
    ];
    Ok(Program {
        graph,
        inputs: [samples, w1, w2, w3, targets],
        outputs: [updated[0], updated[1], updated[2], loss],
    })
}

fn expected(
    program: &Program,
    initial: &[Tensor; 5],
) -> Result<native::MlpWeights, Box<dyn Error>> {
    let mut inputs = initial.clone();
    let mut losses = [0.0; 2];
    for loss in &mut losses {
        let bindings = program
            .inputs
            .into_iter()
            .zip(
                inputs
                    .each_ref()
                    .map(|tensor| TensorValue::F32(tensor.clone())),
            )
            .collect::<Vec<_>>();
        let execution = program.graph.evaluate(&bindings)?;
        for (weight, output) in inputs[1..4].iter_mut().zip(program.outputs[..3].iter()) {
            *weight = execution.value_typed::<f32>(*output)?.clone();
        }
        *loss = execution.value_typed::<f32>(program.outputs[3])?.data()[0];
    }
    Ok(native::MlpWeights {
        w1: inputs[1].data().to_vec(),
        w2: inputs[2].data().to_vec(),
        w3: inputs[3].data().to_vec(),
        losses,
    })
}

fn small_check(checked: bool) -> Result<(), Box<dyn Error>> {
    let program = program(2, 3, 4, 2)?;
    let samples = vec![1.0; 6];
    let targets = vec![0.25; 4];
    let initial = [
        Tensor::new([2, 3], samples.clone())?,
        Tensor::new([3, 4], vec![0.5; 12])?,
        Tensor::new([4, 4], vec![0.25; 16])?,
        Tensor::new([4, 2], vec![0.5; 8])?,
        Tensor::new([2, 2], targets.clone())?,
    ];
    let expected = expected(&program, &initial)?;
    let weights = initial[1..4]
        .iter()
        .map(|weight| weight.data().to_vec())
        .collect::<Vec<_>>();
    let owners = || -> Result<[PcuTensor<f32>; 3], Box<dyn Error>> {
        Ok([
            source::retain(matrix::<3, 4>(&weights[0]).as_ref())?,
            source::retain(matrix::<4, 4>(&weights[1]).as_ref())?,
            source::retain(matrix::<4, 2>(&weights[2]).as_ref())?,
        ])
    };
    let sources = Sources {
        samples: source::retain(matrix::<2, 3>(&samples).as_ref())?,
        targets: source::retain(matrix::<2, 2>(&targets).as_ref())?,
        weights: [owners()?, owners()?],
    };
    let actual = sources.two::<2, 3, 4, 2>(0, checked)?;
    verify_weights(&expected, &actual)?;
    for ((expected, actual), initial) in [
        (&expected.w1, &actual.w1),
        (&expected.w2, &actual.w2),
        (&expected.w3, &actual.w3),
    ]
    .into_iter()
    .zip(&weights)
    {
        assert!(
            expected
                .iter()
                .zip(initial)
                .any(|(updated, old)| (updated - old).abs() > 1.0e-6),
            "small fixture must observe every SGD weight update"
        );
        assert!(
            expected
                .iter()
                .zip(actual)
                .all(|(expected, actual)| (expected - actual).abs() < 1.0e-6),
            "small tuple weights must match the CPU oracle closely"
        );
        if checked {
            assert!(
                expected
                    .iter()
                    .zip(actual)
                    .all(|(expected, actual)| expected.to_bits() == actual.to_bits()),
                "strict small tuple weights match the ordered CPU graph bitwise"
            );
        }
    }
    if checked {
        assert_eq!(
            expected.losses.map(f32::to_bits),
            actual.losses.map(f32::to_bits),
            "strict small tuple losses match the ordered CPU graph bitwise"
        );
    }
    println!("Small source tuple MLP agrees with two-step CPU union-gradient graph");
    Ok(())
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    if std::env::var_os("FUSION_PCU_MLP_CPU_SMALL").is_some() {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Cpu,
            ..global::PcuExecutionPolicy::default()
        })?;
        small_check(true)?;
        println!("CPU small MLP diagnostic only; no GPU execution or latency estimates");
        return Ok(());
    }
    activity::guard();
    let discovery = RocmDiscovery::new();
    let candidates = support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect();
    let (backend, selected) = support::selection::open_ranked(&discovery, candidates, 128)?;
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(selected.device.id),
        block_size: 128,
        score_invocation: Some(score),
        ..global::PcuExecutionPolicy::default()
    })?;
    small_check(false)?;
    large(criterion, &discovery, &backend, &selected)
}

#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // Keep all three complete training boundaries and their owner lifetimes together.
fn large(
    criterion: &mut Criterion,
    discovery: &RocmDiscovery,
    backend: &RocmOwnedDispatchBackend,
    selected: &support::selection::Candidate,
) -> Result<(), Box<dyn Error>> {
    const B: usize = 256;
    const I: usize = 1024;
    const H: usize = 2048;
    const O: usize = 1024;
    activity::guard();
    let samples = values(B * I, 31, 1.0 / 64.0, 15);
    let targets = values(B * O, 19, 1.0 / 32.0, 9);
    let initial = [
        values(I * H, 23, 1.0 / 256.0, 11),
        values(H * H, 23, 1.0 / 256.0, 11),
        values(H * O, 23, 1.0 / 256.0, 11),
    ];
    let mut changed = initial.clone();
    for weight in &mut changed {
        weight[0] += 0.125;
    }
    let weights = [initial, changed];
    let sources = Sources {
        samples: source::retain(matrix::<B, I>(&samples).as_ref())?,
        targets: source::retain(matrix::<B, O>(&targets).as_ref())?,
        weights: weights
            .each_ref()
            .map(|bank| {
                Ok::<_, Box<dyn Error>>([
                    source::retain(matrix::<I, H>(&bank[0]).as_ref())?,
                    source::retain(matrix::<H, H>(&bank[1]).as_ref())?,
                    source::retain(matrix::<H, O>(&bank[2]).as_ref())?,
                ])
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "two source banks required")?,
    };
    let mut natives = weights
        .each_ref()
        .map(|bank| {
            native::NativeMlpTrain::prepare(
                discovery,
                selected.device,
                &native::TrainInputs {
                    batch: B,
                    samples: &samples,
                    targets: &targets,
                    initial_w1: &bank[0],
                    initial_w2: &bank[1],
                    initial_w3: &bank[2],
                    learning_rate: RATE,
                },
            )
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let program = program(B, I, H, O)?;
    let assessor = RocmTensorAssessor::new(backend)?;
    let prepared = assessor.prepare_owned_program(program.graph.into_selected_program(
        &program.outputs,
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?)?;
    let mut memory = backend.memory_provider(selected.pool);
    let device_samples =
        PcuDeviceTensor::new([B, I], backend.upload_buffer(selected.pool, &samples)?)?;
    let device_targets =
        PcuDeviceTensor::new([B, O], backend.upload_buffer(selected.pool, &targets)?)?;
    let device_weights = weights
        .each_ref()
        .map(|bank| {
            Ok::<_, Box<dyn Error>>([
                PcuDeviceTensor::new([I, H], backend.upload_buffer(selected.pool, &bank[0])?)?,
                PcuDeviceTensor::new([H, H], backend.upload_buffer(selected.pool, &bank[1])?)?,
                PcuDeviceTensor::new([H, O], backend.upload_buffer(selected.pool, &bank[2])?)?,
            ])
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let samples_ref = assessor.borrow_device_input_ref(&device_samples, selected.pool)?;
    let targets_ref = assessor.borrow_device_input_ref(&device_targets, selected.pool)?;
    let mut explicit = |bank: usize| -> Result<native::MlpWeights, Box<dyn Error>> {
        let initial = &device_weights[bank];
        let weight_refs = [
            assessor.borrow_device_input_ref(&initial[0], selected.pool)?,
            assessor.borrow_device_input_ref(&initial[1], selected.pool)?,
            assessor.borrow_device_input_ref(&initial[2], selected.pool)?,
        ];
        let first_inputs = [
            (program.inputs[0], &samples_ref),
            (program.inputs[1], &weight_refs[0]),
            (program.inputs[2], &weight_refs[1]),
            (program.inputs[3], &weight_refs[2]),
            (program.inputs[4], &targets_ref),
        ];
        let first = assessor.execute_owned_program_output_array_from_inputs::<f32, _, 4>(
            &prepared,
            &first_inputs,
            selected.pool,
            &mut memory,
        )?;
        let first_refs = [
            assessor.borrow_device_input_ref(&first[0], selected.pool)?,
            assessor.borrow_device_input_ref(&first[1], selected.pool)?,
            assessor.borrow_device_input_ref(&first[2], selected.pool)?,
        ];
        let second_inputs = [
            (program.inputs[0], &samples_ref),
            (program.inputs[1], &first_refs[0]),
            (program.inputs[2], &first_refs[1]),
            (program.inputs[3], &first_refs[2]),
            (program.inputs[4], &targets_ref),
        ];
        let second = assessor.execute_owned_program_output_array_from_inputs::<f32, _, 4>(
            &prepared,
            &second_inputs,
            selected.pool,
            &mut memory,
        )?;
        let read = |owner: &PcuDeviceTensor<f32, _>, elements: usize| {
            let mut output = vec![0.0; elements];
            backend.download_buffer(selected.pool, owner.buffer(), &mut output)?;
            Ok::<_, Box<dyn Error>>(output)
        };
        let mut first_loss = [0.0];
        let mut second_loss = [0.0];
        backend.download_buffer(selected.pool, first[3].buffer(), &mut first_loss)?;
        backend.download_buffer(selected.pool, second[3].buffer(), &mut second_loss)?;
        Ok(native::MlpWeights {
            w1: read(&second[0], I * H)?,
            w2: read(&second[1], H * H)?,
            w3: read(&second[2], H * O)?,
            losses: [first_loss[0], second_loss[0]],
        })
    };
    let mut expected = Vec::with_capacity(2);
    for (bank, native) in natives.iter_mut().enumerate() {
        let result = native.execute_two_steps()?;
        verify_weights(&result, &sources.two::<B, I, H, O>(bank, false)?)?;
        verify_weights(&result, &explicit(bank)?)?;
        expected.push(result);
    }
    assert!(
        (expected[0].w1[0] - expected[1].w1[0]).abs() > 0.05
            && (expected[0].w2[0] - expected[1].w2[0]).abs() > 0.05
            && (expected[0].w3[0] - expected[1].w3[0]).abs() > 0.05,
        "changing banks must expose stale initial-weight bindings"
    );
    println!(
        "MLP {I}->{H}->{H}->{O}, batch {B}, two steps: all three updated weights and both pre-update losses verified for two changing initial banks"
    );
    println!(
        "Source and explicit IR retain immutable starting owners and publish fresh escaped weight/loss owners per step. Native restores its resident ping-pong banks. Timed routes include two training steps and final three-weight/two-loss readback; cold uploads/preparation excluded. This records their actual storage costs."
    );
    let scores = SCORES.load(Ordering::Relaxed);
    let mut phase = 0;
    let mut group = criterion.benchmark_group("rocm_mlp_1024_2048_2048_1024_batch256_two_steps");
    group.throughput(Throughput::Elements(u64::try_from(2 * B)?));
    group.bench_function("macro_source_fresh_owners", |bench| {
        bench.iter(|| {
            phase ^= 1;
            std::hint::black_box(sources.two::<B, I, H, O>(phase, false).unwrap());
        });
    });
    group.bench_function("explicit_ir_fresh_owners", |bench| {
        bench.iter(|| {
            phase ^= 1;
            std::hint::black_box(explicit(phase).unwrap());
        });
    });
    group.bench_function("native_resident_banks", |bench| {
        bench.iter(|| {
            phase ^= 1;
            std::hint::black_box(natives[phase].execute_two_steps().unwrap());
        });
    });
    group.finish();
    assert_eq!(
        SCORES.load(Ordering::Relaxed),
        scores,
        "warm source calls reuse admission"
    );
    // Revisit every bank after warm execution to prove immutable old owners survived feedback.
    for (bank, expected) in expected.iter().enumerate() {
        verify_weights(expected, &sources.two::<B, I, H, O>(bank, false)?)?;
        verify_weights(expected, &explicit(bank)?)?;
    }
    Ok(())
}

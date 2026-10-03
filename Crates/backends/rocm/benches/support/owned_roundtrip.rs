//! Owned upload, PCU dispatch, and fresh host-readback benchmark.
//!
//! The manual HIP control uses the same prepared PCU copy kernel and geometry, streams, resident
//! buffers, and Arc payload. It is a scheduler control, not independent kernel code generation.

#[rustfmt::skip]
use std::{
    error::Error,
    num::NonZeroU32,
    sync::Arc,
    time::{
        Duration,
        Instant,
    },
};

#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    model::dispatch::{
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchFeatureCaps,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchValueId,
    },
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchSubmission,
    PcuExecutionNodeState,
    PcuInvocationShape,
    PcuKernelId,
    PcuOwnedBinding,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipBatchCompletion,
    HipCompletionBatch,
    HipKernelArgument,
    HipStreamHandle,
    RocmExecutionStep,
    RocmOwnedDispatchBackend,
    RocmOwnedExecution,
    RocmOwnedExecutionNode,
    RocmOwnedExecutionOperation,
    RocmOwnedExecutionTwoSlot,
    RocmPreparedDispatch,
    RocmTwoSlotExecutionStep,
};

const COPY_BINDINGS: [PcuBinding<'static>; 2] = [
    PcuBinding::value(
        Some("roundtrip_input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::u32(),
    ),
    PcuBinding::value(
        Some("roundtrip_output"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::u32(),
    ),
];

const COPY_OPS: [PcuDispatchOp<'static>; 3] = [
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: PcuDispatchValueId(1),
        binding: PcuBindingRef::new(0, 0),
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: PcuBindingRef::new(0, 1),
        index: PcuDispatchIndex::InvocationId,
        value: PcuDispatchValueId(1),
    }),
    PcuDispatchOp::Control(fusion_pcu::model::dispatch::PcuDispatchControlOp::Return),
];

struct Roundtrip {
    source: Arc<[u8]>,
    expected: Vec<u8>,
    input: DeviceBuffer,
    output: DeviceBuffer,
    upload_stream: HipStreamHandle,
    readback_stream: HipStreamHandle,
    prepared: RocmPreparedDispatch,
    bindings: [PcuOwnedBinding<DeviceBuffer>; 2],
}

struct PhasedRoundtrip {
    output: Box<[u8]>,
    submission: Duration,
    final_wait: Duration,
}

impl PhasedRoundtrip {
    fn total(&self) -> Duration {
        self.submission + self.final_wait
    }
}

impl Roundtrip {
    fn new(backend: &RocmOwnedDispatchBackend, elements: usize) -> Result<Self, Box<dyn Error>> {
        let expected = (0..elements)
            .map(|index| {
                u32::try_from(index)
                    .expect("bounded roundtrip extent")
                    .wrapping_mul(29)
                    .wrapping_add(11)
            })
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        let source = Arc::<[u8]>::from(expected.clone());
        let input = backend.allocate(expected.len())?;
        let output = backend.allocate(expected.len())?;
        let upload_stream = backend.create_stream()?;
        let readback_stream = backend.create_stream()?;
        let kernel = PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(0xD1_0054),
            entry: PcuDispatchEntryPoint {
                name: "owned_roundtrip_copy",
                logical_shape: [u32::try_from(elements)?, 1, 1],
            },
            bindings: &COPY_BINDINGS,
            ports: &[],
            parameters: &[],
            ops: &COPY_OPS,
            type_caps: PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES,
            feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
                .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
        };
        let prepared = backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(
                NonZeroU32::new(u32::try_from(elements)?).ok_or("zero roundtrip extent")?,
            ),
        })?;
        let bindings = [
            backend.binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::u32()),
                input.clone(),
            )?,
            backend.binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::u32()),
                output.clone(),
            )?,
        ];
        Ok(Self {
            source,
            expected,
            input,
            output,
            upload_stream,
            readback_stream,
            prepared,
            bindings,
        })
    }

    fn run_pcu(&self) -> Result<Box<[u8]>, Box<dyn Error>> {
        let dependencies: [&[usize]; 3] = [&[], &[0], &[1]];
        let nodes = [
            RocmOwnedExecutionNode {
                dependencies: dependencies[0],
                operation: RocmOwnedExecutionOperation::HostUpload {
                    stream: &self.upload_stream,
                    source: Arc::clone(&self.source),
                    destination: &self.input,
                    offset: 0,
                },
            },
            RocmOwnedExecutionNode {
                dependencies: dependencies[1],
                operation: RocmOwnedExecutionOperation::Dispatch {
                    prepared: &self.prepared,
                    bindings: &self.bindings,
                },
            },
            RocmOwnedExecutionNode {
                dependencies: dependencies[2],
                operation: RocmOwnedExecutionOperation::DeviceReadback {
                    stream: &self.readback_stream,
                    source: &self.output,
                    source_offset: 0,
                    bytes: self.expected.len(),
                },
            },
        ];
        let mut scratch = [false; 3];
        let mut states = [PcuExecutionNodeState::Pending; 3];
        let mut execution = RocmOwnedExecution::new(&nodes, &[], &mut scratch, &mut states)?;
        for node in 0..3 {
            if execution.step()? != (RocmExecutionStep::Succeeded { node }) {
                return Err(format!("PCU roundtrip node {node} did not succeed serially").into());
            }
        }
        if execution.step()? != RocmExecutionStep::Complete {
            return Err(format!("PCU roundtrip terminal states were {states:?}").into());
        }
        let output = execution.take_readback(2)?;
        drop(execution);
        if states != [PcuExecutionNodeState::Succeeded; 3] {
            return Err(format!("PCU roundtrip terminal states were {states:?}").into());
        }
        Ok(output)
    }

    fn run_pcu_event_chain(&self) -> Result<Box<[u8]>, Box<dyn Error>> {
        let dependencies: [&[usize]; 3] = [&[], &[0], &[1]];
        let nodes = [
            RocmOwnedExecutionNode {
                dependencies: dependencies[0],
                operation: RocmOwnedExecutionOperation::HostUpload {
                    stream: &self.upload_stream,
                    source: Arc::clone(&self.source),
                    destination: &self.input,
                    offset: 0,
                },
            },
            RocmOwnedExecutionNode {
                dependencies: dependencies[1],
                operation: RocmOwnedExecutionOperation::Dispatch {
                    prepared: &self.prepared,
                    bindings: &self.bindings,
                },
            },
            RocmOwnedExecutionNode {
                dependencies: dependencies[2],
                operation: RocmOwnedExecutionOperation::DeviceReadback {
                    stream: &self.readback_stream,
                    source: &self.output,
                    source_offset: 0,
                    bytes: self.expected.len(),
                },
            },
        ];
        let mut scratch = [false; 3];
        let mut states = [PcuExecutionNodeState::Pending; 3];
        let mut execution =
            RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?;

        // Every node must queue into the event chain before any host completion wait.
        for node in 0..3 {
            if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node }) {
                return Err(
                    format!("PCU event chain did not submit node {node} before waiting").into(),
                );
            }
        }
        if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 2 })
            || execution.step()? != RocmTwoSlotExecutionStep::Complete
        {
            return Err("PCU event-chained roundtrip did not complete successfully".into());
        }
        let output = execution.take_readback(2)?;
        drop(execution);
        if states != [PcuExecutionNodeState::Succeeded; 3] {
            return Err(format!("PCU event-chain states were {states:?}").into());
        }
        Ok(output)
    }

    fn run_pcu_event_chain_phased(&self) -> Result<PhasedRoundtrip, Box<dyn Error>> {
        let submission_start = Instant::now();
        let dependencies: [&[usize]; 3] = [&[], &[0], &[1]];
        let nodes = [
            RocmOwnedExecutionNode {
                dependencies: dependencies[0],
                operation: RocmOwnedExecutionOperation::HostUpload {
                    stream: &self.upload_stream,
                    source: Arc::clone(&self.source),
                    destination: &self.input,
                    offset: 0,
                },
            },
            RocmOwnedExecutionNode {
                dependencies: dependencies[1],
                operation: RocmOwnedExecutionOperation::Dispatch {
                    prepared: &self.prepared,
                    bindings: &self.bindings,
                },
            },
            RocmOwnedExecutionNode {
                dependencies: dependencies[2],
                operation: RocmOwnedExecutionOperation::DeviceReadback {
                    stream: &self.readback_stream,
                    source: &self.output,
                    source_offset: 0,
                    bytes: self.expected.len(),
                },
            },
        ];
        let mut scratch = [false; 3];
        let mut states = [PcuExecutionNodeState::Pending; 3];
        let mut execution =
            RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?;
        for node in 0..3 {
            if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node }) {
                return Err(
                    format!("PCU event chain did not submit node {node} before waiting").into(),
                );
            }
        }
        let submission = submission_start.elapsed();

        let wait_start = Instant::now();
        if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 2 })
            || execution.step()? != RocmTwoSlotExecutionStep::Complete
        {
            return Err("PCU event-chained roundtrip did not complete successfully".into());
        }
        let output = execution.take_readback(2)?;
        drop(execution);
        let final_wait = wait_start.elapsed();
        if states != [PcuExecutionNodeState::Succeeded; 3] {
            return Err(format!("PCU event-chain states were {states:?}").into());
        }
        Ok(PhasedRoundtrip {
            output,
            submission,
            final_wait,
        })
    }

    #[allow(unsafe_code)] // Same prepared copy kernel, bindings, and geometry preserve its ABI.
    fn run_manual_hip(&self) -> Result<Box<[u8]>, Box<dyn Error>> {
        let mut upload = HipCompletionBatch::new(&self.upload_stream);
        upload.copy_host_to_device_at(&self.input, 0, Arc::clone(&self.source))?;
        upload.finish()?.wait()?;

        let (grid, block) = self.prepared.launch_geometry();
        let kernel = self.prepared.hip_kernel();
        let dispatch_stream = self.prepared.stream_handle();
        let arguments = [
            HipKernelArgument::Buffer(&self.input),
            HipKernelArgument::Buffer(&self.output),
        ];
        let mut dispatch = HipCompletionBatch::new(&dispatch_stream);
        // SAFETY: this is the prepared two-buffer copy kernel, using its exact ABI and geometry.
        unsafe {
            kernel.launch_into_batch(&mut dispatch, grid, block, 0, &arguments)?;
        }
        dispatch.finish()?.wait()?;

        let mut readback_batch = HipCompletionBatch::new(&self.readback_stream);
        let readback = readback_batch.allocate_readback(self.expected.len())?;
        readback_batch.copy_device_to_host(&self.output, &readback, self.expected.len())?;
        let mut completion: HipBatchCompletion = readback_batch.finish()?;
        completion.wait()?;
        Ok(completion.take_readback(&readback)?)
    }

    #[allow(unsafe_code)] // Same prepared copy kernel, bindings, and geometry preserve its ABI.
    fn run_manual_hip_event_chain(&self) -> Result<Box<[u8]>, Box<dyn Error>> {
        let mut upload = HipCompletionBatch::new(&self.upload_stream);
        upload.copy_host_to_device_at(&self.input, 0, Arc::clone(&self.source))?;
        let mut upload_completion = upload.finish()?;

        let (grid, block) = self.prepared.launch_geometry();
        let kernel = self.prepared.hip_kernel();
        let dispatch_stream = self.prepared.stream_handle();
        let arguments = [
            HipKernelArgument::Buffer(&self.input),
            HipKernelArgument::Buffer(&self.output),
        ];
        let mut dispatch = HipCompletionBatch::new(&dispatch_stream);
        dispatch.wait_for_batch(&mut upload_completion)?;
        // SAFETY: this is the prepared two-buffer copy kernel, using its exact ABI and geometry.
        unsafe {
            kernel.launch_into_batch(&mut dispatch, grid, block, 0, &arguments)?;
        }
        let mut dispatch_completion = dispatch.finish()?;

        let mut readback_batch = HipCompletionBatch::new(&self.readback_stream);
        readback_batch.wait_for_batch(&mut dispatch_completion)?;
        let readback = readback_batch.allocate_readback(self.expected.len())?;
        readback_batch.copy_device_to_host(&self.output, &readback, self.expected.len())?;
        let mut completion: HipBatchCompletion = readback_batch.finish()?;
        completion.wait()?;
        Ok(completion.take_readback(&readback)?)
    }

    #[allow(unsafe_code)] // Same prepared copy kernel, bindings, and geometry preserve its ABI.
    fn run_manual_hip_event_chain_phased(&self) -> Result<PhasedRoundtrip, Box<dyn Error>> {
        let submission_start = Instant::now();
        let mut upload = HipCompletionBatch::new(&self.upload_stream);
        upload.copy_host_to_device_at(&self.input, 0, Arc::clone(&self.source))?;
        let mut upload_completion = upload.finish()?;

        let (grid, block) = self.prepared.launch_geometry();
        let kernel = self.prepared.hip_kernel();
        let dispatch_stream = self.prepared.stream_handle();
        let arguments = [
            HipKernelArgument::Buffer(&self.input),
            HipKernelArgument::Buffer(&self.output),
        ];
        let mut dispatch = HipCompletionBatch::new(&dispatch_stream);
        dispatch.wait_for_batch(&mut upload_completion)?;
        // SAFETY: this is the prepared two-buffer copy kernel, using its exact ABI and geometry.
        unsafe {
            kernel.launch_into_batch(&mut dispatch, grid, block, 0, &arguments)?;
        }
        let mut dispatch_completion = dispatch.finish()?;

        let mut readback_batch = HipCompletionBatch::new(&self.readback_stream);
        readback_batch.wait_for_batch(&mut dispatch_completion)?;
        let readback = readback_batch.allocate_readback(self.expected.len())?;
        readback_batch.copy_device_to_host(&self.output, &readback, self.expected.len())?;
        let mut completion: HipBatchCompletion = readback_batch.finish()?;
        let submission = submission_start.elapsed();

        let wait_start = Instant::now();
        completion.wait()?;
        let output = completion.take_readback(&readback)?;
        let final_wait = wait_start.elapsed();
        Ok(PhasedRoundtrip {
            output,
            submission,
            final_wait,
        })
    }

    fn verify(&self, route: &str, actual: &[u8]) -> Result<(), Box<dyn Error>> {
        if actual != self.expected {
            return Err(format!("{route} roundtrip output mismatch").into());
        }
        Ok(())
    }

    fn paired_event_diagnostic(&self, elements: usize) -> Result<(), Box<dyn Error>> {
        const PAIRS: usize = 16;
        const SAMPLES_PER_ROUTE: usize = 5;
        let mut pcu_pairs = Vec::with_capacity(PAIRS);
        let mut hip_pairs = Vec::with_capacity(PAIRS);
        let mut ratios = Vec::with_capacity(PAIRS);

        for pair in 0..PAIRS {
            let pcu_first = pair.is_multiple_of(2);
            let mut pcu_samples = Vec::with_capacity(SAMPLES_PER_ROUTE);
            let mut hip_samples = Vec::with_capacity(SAMPLES_PER_ROUTE);
            let mut pcu_submission = Vec::with_capacity(SAMPLES_PER_ROUTE);
            let mut pcu_wait = Vec::with_capacity(SAMPLES_PER_ROUTE);
            let mut hip_submission = Vec::with_capacity(SAMPLES_PER_ROUTE);
            let mut hip_wait = Vec::with_capacity(SAMPLES_PER_ROUTE);

            for sample in 0..SAMPLES_PER_ROUTE {
                let pcu_first_this_sample = if sample.is_multiple_of(2) {
                    pcu_first
                } else {
                    !pcu_first
                };
                for use_pcu in [pcu_first_this_sample, !pcu_first_this_sample] {
                    let phased = if use_pcu {
                        self.run_pcu_event_chain_phased()?
                    } else {
                        self.run_manual_hip_event_chain_phased()?
                    };
                    self.verify(
                        if use_pcu {
                            "PCU event-chain paired diagnostic"
                        } else {
                            "manual HIP event-chain paired diagnostic"
                        },
                        &phased.output,
                    )?;
                    if use_pcu {
                        pcu_samples.push(phased.total().as_secs_f64() * 1.0e6);
                        pcu_submission.push(phased.submission.as_secs_f64() * 1.0e6);
                        pcu_wait.push(phased.final_wait.as_secs_f64() * 1.0e6);
                    } else {
                        hip_samples.push(phased.total().as_secs_f64() * 1.0e6);
                        hip_submission.push(phased.submission.as_secs_f64() * 1.0e6);
                        hip_wait.push(phased.final_wait.as_secs_f64() * 1.0e6);
                    }
                }
            }

            let pcu_median = median(&mut pcu_samples);
            let hip_median = median(&mut hip_samples);
            let ratio = pcu_median / hip_median;
            ratios.push(ratio);
            pcu_pairs.push((
                pcu_median,
                median(&mut pcu_submission),
                median(&mut pcu_wait),
            ));
            hip_pairs.push((
                hip_median,
                median(&mut hip_submission),
                median(&mut hip_wait),
            ));
            println!(
                "roundtrip paired {elements} pair={:02} order={} PCU median={pcu_median:.3}us (submit {:.3}us, final wait {:.3}us) manual HIP median={hip_median:.3}us (submit {:.3}us, final wait {:.3}us) ratio={ratio:.4}",
                pair + 1,
                if pcu_first { "PCU-first" } else { "HIP-first" },
                pcu_pairs[pair].1,
                pcu_pairs[pair].2,
                hip_pairs[pair].1,
                hip_pairs[pair].2,
            );
        }

        let pcu_total = pcu_pairs.iter().map(|sample| sample.0).collect::<Vec<_>>();
        let pcu_submit = pcu_pairs.iter().map(|sample| sample.1).collect::<Vec<_>>();
        let pcu_wait = pcu_pairs.iter().map(|sample| sample.2).collect::<Vec<_>>();
        let hip_total = hip_pairs.iter().map(|sample| sample.0).collect::<Vec<_>>();
        let hip_submit = hip_pairs.iter().map(|sample| sample.1).collect::<Vec<_>>();
        let hip_wait = hip_pairs.iter().map(|sample| sample.2).collect::<Vec<_>>();
        println!(
            "roundtrip paired summary {elements}: PCU total {}; submit {}; final wait {}; manual HIP total {}; submit {}; final wait {}; paired PCU/HIP ratio {}",
            summarize(&pcu_total),
            summarize(&pcu_submit),
            summarize(&pcu_wait),
            summarize(&hip_total),
            summarize(&hip_submit),
            summarize(&hip_wait),
            summarize_ratio(&ratios),
        );
        Ok(())
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        f64::midpoint(samples[middle - 1], samples[middle])
    } else {
        samples[middle]
    }
}

fn summarize(samples: &[f64]) -> String {
    let mut ordered = samples.to_vec();
    let mid = median(&mut ordered);
    let low = ordered.first().copied().unwrap_or_default();
    let high = ordered.last().copied().unwrap_or_default();
    format!("median {mid:.3}us (range {low:.3}-{high:.3}us)")
}

fn summarize_ratio(samples: &[f64]) -> String {
    let mut ordered = samples.to_vec();
    let mid = median(&mut ordered);
    let low = ordered.first().copied().unwrap_or_default();
    let high = ordered.last().copied().unwrap_or_default();
    format!("median {mid:.3}x (range {low:.3}-{high:.3}x)")
}

/// Compare the owned serial graph with a manual HIP scheduler control using identical GPU work.
pub fn benchmark(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    println!(
        "owned HostUpload -> Dispatch -> DeviceReadback roundtrip: compares serial and event-chained routes; manual HIP uses the same prepared PCU kernel/geometry, streams, resident buffers, and Arc payload as a scheduler control, not independent codegen"
    );
    let reverse_order = std::env::var_os("FUSION_ROCM_HOST_ROUNDTRIP_REVERSE_ORDER").as_deref()
        == Some(std::ffi::OsStr::new("1"));
    for elements in [65, 1 << 20] {
        let roundtrip = Roundtrip::new(backend, elements)?;
        // Preparation and device allocations are outside timed calls. Each route gets an exact
        // readback check before Criterion, including its own newly allocated host readback.
        let pcu_output = roundtrip.run_pcu()?;
        roundtrip.verify("PCU serial graph", &pcu_output)?;
        let manual_output = roundtrip.run_manual_hip()?;
        roundtrip.verify("manual HIP serial control", &manual_output)?;
        let pcu_event_output = roundtrip.run_pcu_event_chain()?;
        roundtrip.verify("PCU event-chain graph", &pcu_event_output)?;
        let manual_event_output = roundtrip.run_manual_hip_event_chain()?;
        roundtrip.verify("manual HIP event-chain control", &manual_event_output)?;
        if std::env::var_os("FUSION_ROCM_HOST_ROUNDTRIP_PAIRED_DIAGNOSTIC").as_deref()
            == Some(std::ffi::OsStr::new("1"))
        {
            roundtrip.paired_event_diagnostic(elements)?;
        }

        let mut group = criterion.benchmark_group(format!("owned_roundtrip/{elements}"));
        group.throughput(Throughput::Bytes(u64::try_from(roundtrip.expected.len())?));
        let routes = if reverse_order {
            [
                ("manual HIP event-chain control", false, true),
                ("PCU event-chain graph", true, true),
                ("manual HIP serial control", false, false),
                ("PCU serial graph", true, false),
            ]
        } else {
            [
                ("PCU serial graph", true, false),
                ("manual HIP serial control", false, false),
                ("PCU event-chain graph", true, true),
                ("manual HIP event-chain control", false, true),
            ]
        };
        for (route, use_pcu, event_chain) in routes {
            group.bench_with_input(
                BenchmarkId::new(route, elements),
                &(use_pcu, event_chain),
                |bencher, &(use_pcu, event_chain)| {
                    bencher.iter(|| {
                        let output = if use_pcu && event_chain {
                            roundtrip.run_pcu_event_chain()
                        } else if use_pcu {
                            roundtrip.run_pcu()
                        } else if event_chain {
                            roundtrip.run_manual_hip_event_chain()
                        } else {
                            roundtrip.run_manual_hip()
                        }
                        .expect("owned roundtrip execution failed");
                        std::hint::black_box(output);
                    });
                },
            );
        }
        group.finish();
    }
    Ok(())
}

//! Same-handle dependent SGEMM graph and batch-queue benchmark.
//!
//! The manual route queues the same two rocBLAS SGEMMs into one HIP batch on the same stream.
//! It controls for scheduler cost; it does not compare independent SGEMM implementations.

use std::{
    error::Error,
    time::Instant,
};

use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
};
use fusion_pcu::PcuExecutionNodeState;
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipCompletionBatch,
    HipStreamHandle,
    Rocblas,
    RocmOwnedDispatchBackend,
    RocmOwnedExecutionNode,
    RocmOwnedExecutionOperation,
    RocmOwnedExecutionTwoSlot,
    RocmTwoSlotExecutionStep,
};

const SGEMM_DEPENDENCIES: [&[usize]; 2] = [&[], &[0]];

struct SameHandleSgemmPair {
    dimension: usize,
    left: DeviceBuffer,
    identity: DeviceBuffer,
    intermediate: DeviceBuffer,
    output: DeviceBuffer,
    expected: Vec<u8>,
    stream: HipStreamHandle,
    handle: Rocblas,
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

impl SameHandleSgemmPair {
    fn new(backend: &RocmOwnedDispatchBackend, dimension: usize) -> Result<Self, Box<dyn Error>> {
        let elements = dimension
            .checked_mul(dimension)
            .ok_or("same-handle SGEMM element count overflow")?;
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or("same-handle SGEMM allocation size overflow")?;
        let left_values = (0..elements)
            .map(|index| f32::from(u8::try_from(index % 29 + 1).expect("bounded matrix value")))
            .collect::<Vec<_>>();
        let mut identity_values = vec![0.0_f32; elements];
        for diagonal in 0..dimension {
            identity_values[diagonal * dimension + diagonal] = 1.0;
        }
        let expected = encode_f32(&left_values);
        let identity_bytes = encode_f32(&identity_values);

        let mut left = backend.allocate(bytes)?;
        let mut identity = backend.allocate(bytes)?;
        let intermediate = backend.allocate(bytes)?;
        let output = backend.allocate(bytes)?;
        left.copy_from(&expected)?;
        identity.copy_from(&identity_bytes)?;

        let stream = backend.create_stream()?;
        let mut handle = backend.create_rocblas()?;
        handle.bind_stream(&stream)?;
        Ok(Self {
            dimension,
            left,
            identity,
            intermediate,
            output,
            expected,
            stream,
            handle,
        })
    }

    const fn graph_nodes(&self) -> [RocmOwnedExecutionNode<'_>; 2] {
        [
            RocmOwnedExecutionNode {
                dependencies: SGEMM_DEPENDENCIES[0],
                operation: RocmOwnedExecutionOperation::Sgemm {
                    handle: &self.handle,
                    stream: &self.stream,
                    transpose_a: false,
                    transpose_b: false,
                    m: self.dimension,
                    n: self.dimension,
                    k: self.dimension,
                    alpha: 1.0,
                    a: &self.left,
                    lda: self.dimension,
                    b: &self.identity,
                    ldb: self.dimension,
                    beta: 0.0,
                    c: &self.intermediate,
                    ldc: self.dimension,
                },
            },
            RocmOwnedExecutionNode {
                dependencies: SGEMM_DEPENDENCIES[1],
                operation: RocmOwnedExecutionOperation::Sgemm {
                    handle: &self.handle,
                    stream: &self.stream,
                    transpose_a: false,
                    transpose_b: false,
                    m: self.dimension,
                    n: self.dimension,
                    k: self.dimension,
                    alpha: 1.0,
                    a: &self.intermediate,
                    lda: self.dimension,
                    b: &self.identity,
                    ldb: self.dimension,
                    beta: 0.0,
                    c: &self.output,
                    ldc: self.dimension,
                },
            },
        ]
    }

    fn run_pcu_host_wait(&self) -> Result<(), Box<dyn Error>> {
        let nodes = self.graph_nodes();
        let mut scratch = [false; 2];
        let mut states = [PcuExecutionNodeState::Pending; 2];
        let mut execution = RocmOwnedExecutionTwoSlot::new(&nodes, &[], &mut scratch, &mut states)?;
        let expected_steps = [
            RocmTwoSlotExecutionStep::Submitted { node: 0 },
            RocmTwoSlotExecutionStep::Succeeded { node: 0 },
            RocmTwoSlotExecutionStep::Submitted { node: 1 },
            RocmTwoSlotExecutionStep::Succeeded { node: 1 },
            RocmTwoSlotExecutionStep::Complete,
        ];
        for expected in expected_steps {
            if execution.step()? != expected {
                return Err(format!("same-handle host-wait expected step {expected:?}").into());
            }
        }
        drop(execution);
        if states != [PcuExecutionNodeState::Succeeded; 2] {
            return Err(format!("same-handle host-wait states were {states:?}").into());
        }
        Ok(())
    }

    fn run_pcu_event_chain(&self) -> Result<(), Box<dyn Error>> {
        let nodes = self.graph_nodes();
        let mut scratch = [false; 2];
        let mut states = [PcuExecutionNodeState::Pending; 2];
        let mut execution =
            RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?;
        for node in 0..2 {
            if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node }) {
                return Err(format!(
                    "same-handle event-chain did not queue SGEMM node {node} before a host wait"
                )
                .into());
            }
        }
        if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 1 })
            || execution.step()? != RocmTwoSlotExecutionStep::Complete
        {
            return Err("same-handle SGEMM event chain did not complete".into());
        }
        drop(execution);
        if states != [PcuExecutionNodeState::Succeeded; 2] {
            return Err(format!("same-handle event-chain states were {states:?}").into());
        }
        Ok(())
    }

    fn run_manual_batch_queue(&self) -> Result<(), Box<dyn Error>> {
        let mut batch = HipCompletionBatch::new(&self.stream);
        self.submit_sgemm(&mut batch, &self.left, &self.intermediate)?;
        self.submit_sgemm(&mut batch, &self.intermediate, &self.output)?;
        batch.finish()?.wait()?;
        Ok(())
    }

    fn submit_sgemm(
        &self,
        batch: &mut HipCompletionBatch,
        left: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<(), Box<dyn Error>> {
        self.handle.sgemm_into_batch(
            batch,
            false,
            false,
            self.dimension,
            self.dimension,
            self.dimension,
            1.0,
            left,
            self.dimension,
            &self.identity,
            self.dimension,
            0.0,
            output,
            self.dimension,
        )?;
        Ok(())
    }

    fn verify_output(&self, route: &str) -> Result<(), Box<dyn Error>> {
        let mut actual = vec![0_u8; self.expected.len()];
        self.output.copy_to(&mut actual)?;
        if actual != self.expected {
            return Err(format!(
                "{route} {}x{} output mismatch",
                self.dimension, self.dimension
            )
            .into());
        }
        Ok(())
    }
}

/// Compare two dependent SGEMMs through PCU and a matched one-batch manual rocBLAS queue.
pub fn benchmark(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    println!(
        "same-handle dependent SGEMM: manual HIP + rocBLAS queues both same-stream operations into one batch; this is a scheduler/batching control, not independent rocBLAS codegen"
    );
    let reverse_order = std::env::var_os("FUSION_ROCM_SAME_HANDLE_SGEMM_REVERSE_ORDER").as_deref()
        == Some(std::ffi::OsStr::new("1"));
    for dimension in [2, 1024] {
        let pair = SameHandleSgemmPair::new(backend, dimension)?;
        // Exact-result readback and all allocation/input preparation stay outside Criterion.
        pair.run_pcu_host_wait()?;
        pair.verify_output("PCU host-wait")?;
        pair.run_pcu_event_chain()?;
        pair.verify_output("PCU same-handle event chain")?;
        pair.run_manual_batch_queue()?;
        pair.verify_output("manual one-batch rocBLAS")?;

        let elements = dimension
            .checked_mul(dimension)
            .and_then(|count| count.checked_mul(2))
            .ok_or("same-handle SGEMM throughput overflow")?;
        let mut group =
            criterion.benchmark_group(format!("rocm-same-handle-dependent-sgemm/{dimension}"));
        group.throughput(Throughput::Elements(u64::try_from(elements)?));
        let routes = if reverse_order {
            [
                ("manual HIP rocBLAS one-batch queue", 2_u8),
                ("PCU host-wait fallback", 1),
                ("PCU event-chain queue", 0),
            ]
        } else {
            [
                ("PCU event-chain queue", 0_u8),
                ("PCU host-wait fallback", 1),
                ("manual HIP rocBLAS one-batch queue", 2),
            ]
        };
        for (label, route) in routes {
            group.bench_function(BenchmarkId::new(label, dimension), |bencher| {
                bencher.iter_custom(|iterations| {
                    let start = Instant::now();
                    for _ in 0..iterations {
                        match route {
                            0 => pair
                                .run_pcu_event_chain()
                                .expect("same-handle PCU event-chain run"),
                            1 => pair
                                .run_pcu_host_wait()
                                .expect("same-handle PCU host-wait run"),
                            _ => pair
                                .run_manual_batch_queue()
                                .expect("same-handle manual rocBLAS batch run"),
                        }
                    }
                    start.elapsed()
                });
            });
        }
        group.finish();
    }
    Ok(())
}

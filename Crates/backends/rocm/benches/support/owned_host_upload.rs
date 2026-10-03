//! Criterion comparison of the owned `HostUpload` -> Dispatch graph routes.
//!
//! Both routes reuse one selected backend, prepared copy kernel, device-buffer pair, and
//! reference-counted host payload. This isolates scheduler-route costs; it is not a native HIP
//! parity benchmark.

#[rustfmt::skip]
use std::{
    error::Error,
    num::NonZeroU32,
    sync::Arc,
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
    RocmOwnedDispatchBackend,
    RocmOwnedExecutionNode,
    RocmOwnedExecutionOperation,
    RocmOwnedExecutionTwoSlot,
    RocmTwoSlotExecutionStep,
};

const COPY_BINDINGS: [PcuBinding<'static>; 2] = [
    PcuBinding::value(
        Some("upload_copy_input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::u32(),
    ),
    PcuBinding::value(
        Some("upload_copy_output"),
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

struct UploadGraph {
    source: Arc<[u8]>,
    input: DeviceBuffer,
    output: DeviceBuffer,
    upload_stream: fusion_pcu_rocm::HipStreamHandle,
    prepared: fusion_pcu_rocm::RocmPreparedDispatch,
    bindings: [PcuOwnedBinding<DeviceBuffer>; 2],
    expected: Vec<u8>,
}

impl UploadGraph {
    fn new(backend: &RocmOwnedDispatchBackend, elements: usize) -> Result<Self, Box<dyn Error>> {
        let expected = (0..elements)
            .map(|index| {
                u32::try_from(index)
                    .expect("bounded graph benchmark extent")
                    .wrapping_mul(17)
                    .wrapping_add(23)
            })
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        let source: Arc<[u8]> = Arc::from(expected.clone());
        let input = backend.allocate(expected.len())?;
        let output = backend.allocate(expected.len())?;
        let upload_stream = backend.create_stream()?;
        let kernel = PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(0xD1_0053),
            entry: PcuDispatchEntryPoint {
                name: "owned_host_upload_graph_copy",
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
                NonZeroU32::new(u32::try_from(elements)?).ok_or("zero graph extent")?,
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
            input,
            output,
            upload_stream,
            prepared,
            bindings,
            expected,
        })
    }

    fn run_once(&self, event_chaining: bool) -> Result<(), Box<dyn Error>> {
        let dependencies: [&[usize]; 2] = [&[], &[0]];
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
        ];
        let mut scratch = [false; 2];
        let mut states = [PcuExecutionNodeState::Pending; 2];
        let mut execution = if event_chaining {
            RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?
        } else {
            RocmOwnedExecutionTwoSlot::new(&nodes, &[], &mut scratch, &mut states)?
        };

        if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 0 }) {
            return Err("host-upload graph did not submit upload node".into());
        }
        if event_chaining {
            if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 1 }) {
                return Err("event-chain route did not submit dispatch before waiting".into());
            }
        } else if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 0 })
            || execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 1 })
        {
            return Err("host-wait route did not complete upload before dispatch".into());
        }
        if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 1 })
            || execution.step()? != RocmTwoSlotExecutionStep::Complete
        {
            return Err("host-upload graph did not complete successfully".into());
        }
        if states != [PcuExecutionNodeState::Succeeded; 2] {
            return Err(format!("host-upload graph states were {states:?}").into());
        }
        Ok(())
    }

    fn verify(&self, route: &str) -> Result<(), Box<dyn Error>> {
        let mut actual = vec![0_u8; self.expected.len()];
        self.output.copy_to(&mut actual)?;
        if actual != self.expected {
            return Err(format!("{route} host-upload graph output mismatch").into());
        }
        Ok(())
    }
}

/// Benchmark scheduler route for persistent shared host payload upload followed by a PCU copy.
#[allow(clippy::significant_drop_tightening)] // Criterion's finish consumes and drops the group.
pub fn benchmark(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    println!(
        "owned HostUpload -> Dispatch comparison: same backend, shared Arc payload, resident device buffers, and prepared kernel; compares two-slot host-wait and event-chain scheduler routes, not native HIP parity"
    );
    let reverse_order = std::env::var_os("FUSION_ROCM_HOST_UPLOAD_BENCH_REVERSE_ORDER").as_deref()
        == Some(std::ffi::OsStr::new("1"));
    let mut group = criterion.benchmark_group("owned_host_upload_dispatch_routes");
    for elements in [65, 1 << 20] {
        let graph = UploadGraph::new(backend, elements)?;
        // All compilation, allocations, and upload payload construction precede timing.
        graph.run_once(false)?;
        graph.verify("host-wait preflight")?;
        graph.run_once(true)?;
        graph.verify("event-chain preflight")?;

        let bytes = u64::try_from(graph.expected.len())?;
        group.throughput(Throughput::Bytes(bytes));
        let routes = if reverse_order {
            [("event_chain", true), ("host_wait", false)]
        } else {
            [("host_wait", false), ("event_chain", true)]
        };
        for (route, event_chaining) in routes {
            group.bench_with_input(
                BenchmarkId::new(route, elements),
                &event_chaining,
                |bencher, &event_chaining| {
                    bencher.iter(|| {
                        graph
                            .run_once(event_chaining)
                            .expect("owned host-upload graph execution failed");
                    });
                },
            );
        }
    }
    group.finish();
    Ok(())
}

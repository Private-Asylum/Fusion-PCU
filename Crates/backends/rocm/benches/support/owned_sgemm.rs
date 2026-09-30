//! Criterion comparison for the typed owned dispatch -> SGEMM -> dispatch graph.
//!
//! The manual HIP + rocBLAS peer reuses the exact compiled PCU copy kernel, launch geometry,
//! streams, buffers, and rocBLAS handle. It is a scheduler-overhead control, not independent
//! handwritten kernel codegen.

#[rustfmt::skip]
use std::{
    error::Error,
    num::NonZeroU32,
    time::Instant,
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
    HipCompletion,
    HipCompletionBatch,
    HipKernel,
    HipKernelArgument,
    HipStreamHandle,
    Rocblas,
    RocmOwnedDispatchBackend,
    RocmOwnedExecutionNode,
    RocmOwnedExecutionOperation,
    RocmOwnedExecutionTwoSlot,
    RocmPreparedDispatch,
    RocmTwoSlotExecutionStep,
};

const COPY_BINDINGS: [PcuBinding<'static>; 2] = [
    PcuBinding::value(
        Some("copy_input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::u32(),
    ),
    PcuBinding::value(
        Some("copy_output"),
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

struct SgemmGraph {
    a: DeviceBuffer,
    b: DeviceBuffer,
    c: DeviceBuffer,
    output: DeviceBuffer,
    expected_output: Vec<u8>,
    input_bindings: [PcuOwnedBinding<DeviceBuffer>; 2],
    output_bindings: [PcuOwnedBinding<DeviceBuffer>; 2],
    copy_in: RocmPreparedDispatch,
    copy_out: RocmPreparedDispatch,
    hip_copy_in: HipKernel,
    hip_copy_out: HipKernel,
    copy_in_stream: HipStreamHandle,
    copy_out_stream: HipStreamHandle,
    copy_in_geometry: ([u32; 3], [u32; 3]),
    copy_out_geometry: ([u32; 3], [u32; 3]),
    handle: Rocblas,
    sgemm_stream: fusion_pcu_rocm::HipStreamHandle,
    dimension: usize,
}

impl SgemmGraph {
    #[allow(clippy::too_many_lines)] // Keep resident graph setup and its ownership in one fixture.
    fn new(backend: &RocmOwnedDispatchBackend, dimension: usize) -> Result<Self, Box<dyn Error>> {
        let elements = dimension
            .checked_mul(dimension)
            .ok_or("SGEMM element count overflow")?;
        let byte_len = elements
            .checked_mul(size_of::<f32>())
            .ok_or("SGEMM allocation size overflow")?;
        let mut input_values = vec![0.0_f32; elements];
        for diagonal in 0..dimension {
            input_values[diagonal * dimension + diagonal] = 1.0;
        }
        let expected_output = (0..elements)
            .map(|index| f32::from(u8::try_from(index % 127 + 1).expect("bounded matrix value")))
            .collect::<Vec<_>>();
        let input_bytes = encode_f32(&input_values);
        let expected_bytes = encode_f32(&expected_output);

        let mut input = backend.allocate(byte_len)?;
        let a = backend.allocate(byte_len)?;
        let mut b = backend.allocate(byte_len)?;
        let c = backend.allocate(byte_len)?;
        let output = backend.allocate(byte_len)?;
        input.copy_from(&input_bytes)?;
        b.copy_from(&expected_bytes)?;

        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(0xD1_0052),
            entry: PcuDispatchEntryPoint {
                name: "owned_dispatch_sgemm_copy",
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
        let shape = PcuInvocationShape::invocations(
            NonZeroU32::new(u32::try_from(elements)?).ok_or("zero SGEMM extent")?,
        );
        let copy_in = backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape,
        })?;
        let copy_out = backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape,
        })?;
        let hip_copy_in = copy_in.hip_kernel();
        let hip_copy_out = copy_out.hip_kernel();
        let copy_in_stream = copy_in.stream_handle();
        let copy_out_stream = copy_out.stream_handle();
        let copy_in_geometry = copy_in.launch_geometry();
        let copy_out_geometry = copy_out.launch_geometry();
        let input_bindings = [
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
                a.clone(),
            )?,
        ];
        let output_bindings = [
            backend.binding(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::u32()),
                c.clone(),
            )?,
            backend.binding(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::u32()),
                output.clone(),
            )?,
        ];
        let sgemm_stream = backend.create_stream()?;
        let mut handle = backend.create_rocblas()?;
        handle.bind_stream(&sgemm_stream)?;

        Ok(Self {
            a,
            b,
            c,
            output,
            expected_output: expected_bytes,
            input_bindings,
            output_bindings,
            copy_in,
            copy_out,
            hip_copy_in,
            hip_copy_out,
            copy_in_stream,
            copy_out_stream,
            copy_in_geometry,
            copy_out_geometry,
            handle,
            sgemm_stream,
            dimension,
        })
    }

    fn run_once(&self, event_chaining: bool) -> Result<(), Box<dyn Error>> {
        let dependencies: [&[usize]; 3] = [&[], &[0], &[1]];
        let nodes = [
            RocmOwnedExecutionNode {
                dependencies: dependencies[0],
                operation: RocmOwnedExecutionOperation::Dispatch {
                    prepared: &self.copy_in,
                    bindings: &self.input_bindings,
                },
            },
            RocmOwnedExecutionNode {
                dependencies: dependencies[1],
                operation: RocmOwnedExecutionOperation::Sgemm {
                    handle: &self.handle,
                    stream: &self.sgemm_stream,
                    transpose_a: false,
                    transpose_b: false,
                    m: self.dimension,
                    n: self.dimension,
                    k: self.dimension,
                    alpha: 1.0,
                    a: &self.a,
                    lda: self.dimension,
                    b: &self.b,
                    ldb: self.dimension,
                    beta: 0.0,
                    c: &self.c,
                    ldc: self.dimension,
                },
            },
            RocmOwnedExecutionNode {
                dependencies: dependencies[2],
                operation: RocmOwnedExecutionOperation::Dispatch {
                    prepared: &self.copy_out,
                    bindings: &self.output_bindings,
                },
            },
        ];
        let mut scratch = [false; 3];
        let mut states = [PcuExecutionNodeState::Pending; 3];
        let mut execution = if event_chaining {
            RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?
        } else {
            RocmOwnedExecutionTwoSlot::new(&nodes, &[], &mut scratch, &mut states)?
        };
        if event_chaining {
            for node in 0..3 {
                if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node }) {
                    return Err(
                        format!("event-chain SGEMM graph did not submit node {node}").into(),
                    );
                }
            }
        }
        loop {
            if execution.step()? == RocmTwoSlotExecutionStep::Complete {
                break;
            }
        }
        if states != [PcuExecutionNodeState::Succeeded; 3] {
            return Err(format!("SGEMM graph terminal states were {states:?}").into());
        }
        Ok(())
    }

    /// Manually issue the same compiled HIP copy kernels and rocBLAS SGEMM without constructing
    /// the PCU graph executor. This is a scheduler-overhead control, not separate kernel codegen.
    #[allow(unsafe_code)] // The calls use the exact ABI and geometry captured by preparation.
    fn run_manual_hip_once(&self, event_chaining: bool) -> Result<(), Box<dyn Error>> {
        let input = &self.input_bindings[0].resource;
        let a = &self.input_bindings[1].resource;
        let c = &self.output_bindings[0].resource;
        let output = &self.output_bindings[1].resource;
        let (in_grid, in_block) = self.copy_in_geometry;
        let (out_grid, out_block) = self.copy_out_geometry;

        let producer = launch_native_copy(
            &self.hip_copy_in,
            &self.copy_in_stream,
            in_grid,
            in_block,
            input,
            a,
        )?;
        if !event_chaining {
            let mut producer = producer;
            producer.wait()?;
            let mut sgemm_batch = HipCompletionBatch::new(&self.sgemm_stream);
            self.submit_sgemm(&mut sgemm_batch)?;
            sgemm_batch.finish()?.wait()?;
            let mut consumer = launch_native_copy(
                &self.hip_copy_out,
                &self.copy_out_stream,
                out_grid,
                out_block,
                c,
                output,
            )?;
            consumer.wait()?;
            return Ok(());
        }

        let mut sgemm_batch = HipCompletionBatch::new(&self.sgemm_stream);
        sgemm_batch.wait_for(producer)?;
        self.submit_sgemm(&mut sgemm_batch)?;
        let mut sgemm_completion = sgemm_batch.finish()?;

        let mut copy_out_batch = HipCompletionBatch::new(&self.copy_out_stream);
        copy_out_batch.wait_for_batch(&mut sgemm_completion)?;
        let output_arguments = [
            HipKernelArgument::Buffer(c),
            HipKernelArgument::Buffer(output),
        ];
        // SAFETY: this is the same prepared copy kernel and two-buffer ABI used by the PCU
        // executable; the exact launch geometry and captured stream are reused as well.
        unsafe {
            self.hip_copy_out.launch_into_batch(
                &mut copy_out_batch,
                out_grid,
                out_block,
                0,
                &output_arguments,
            )
        }?;
        copy_out_batch.finish()?.wait()?;
        Ok(())
    }

    fn submit_sgemm(&self, batch: &mut HipCompletionBatch) -> Result<(), Box<dyn Error>> {
        self.handle.sgemm_into_batch(
            batch,
            false,
            false,
            self.dimension,
            self.dimension,
            self.dimension,
            1.0,
            &self.a,
            self.dimension,
            &self.b,
            self.dimension,
            0.0,
            &self.c,
            self.dimension,
        )?;
        Ok(())
    }

    fn verify_output(&self) -> Result<(), Box<dyn Error>> {
        let mut actual = vec![0_u8; self.expected_output.len()];
        self.output.copy_to(&mut actual)?;
        if actual != self.expected_output {
            return Err(format!(
                "{}x{} SGEMM graph output mismatch",
                self.dimension, self.dimension
            )
            .into());
        }
        Ok(())
    }
}

#[allow(unsafe_code)] // Safety contract is documented at the HIP launch call below.
fn launch_native_copy(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    source: &DeviceBuffer,
    destination: &DeviceBuffer,
) -> Result<HipCompletion, fusion_pcu_rocm::HipError> {
    let arguments = [
        HipKernelArgument::Buffer(source),
        HipKernelArgument::Buffer(destination),
    ];
    // SAFETY: this helper is called only with the prepared two-buffer copy kernel and exact
    // geometry. The HIP completion token retains the kernel module, stream, and both allocations.
    unsafe { kernel.launch(stream, grid, block, 0, &arguments) }
}

/// Register orchestration-small and workload-heavy resident-resource comparisons.
pub fn benchmark(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    for dimension in [2, 1024] {
        let graph = SgemmGraph::new(backend, dimension)?;
        graph.run_once(false)?;
        graph.verify_output()?;
        graph.run_once(true)?;
        graph.verify_output()?;
        graph.run_manual_hip_once(false)?;
        graph.verify_output()?;
        graph.run_manual_hip_once(true)?;
        graph.verify_output()?;

        {
            let mut group = criterion.benchmark_group(format!(
                "rocm-owned-dispatch-sgemm-dispatch/{dimension}x{dimension}"
            ));
            group.throughput(Throughput::Elements((dimension * dimension) as u64));
            let routes = if std::env::var_os("FUSION_ROCM_SGEMM_BENCH_REVERSE_ORDER").as_deref()
                == Some(std::ffi::OsStr::new("1"))
            {
                [
                    ("manual HIP event chain", true, true),
                    ("manual HIP host wait", false, true),
                    ("PCU event chain", true, false),
                    ("PCU two-slot host wait", false, false),
                ]
            } else {
                [
                    ("PCU two-slot host wait", false, false),
                    ("PCU event chain", true, false),
                    ("manual HIP host wait", false, true),
                    ("manual HIP event chain", true, true),
                ]
            };
            for (label, event_chaining, manual_hip) in routes {
                group.bench_function(BenchmarkId::new(label, dimension), |bencher| {
                    bencher.iter_custom(|iterations| {
                        let started = Instant::now();
                        for _ in 0..iterations {
                            if manual_hip {
                                graph
                                    .run_manual_hip_once(event_chaining)
                                    .expect("resident manual HIP SGEMM graph iteration");
                            } else {
                                graph
                                    .run_once(event_chaining)
                                    .expect("resident PCU SGEMM graph iteration");
                            }
                        }
                        started.elapsed()
                    });
                });
            }
            group.finish();
        }
        // Readback and comparison are kept outside the measured iterations.
        graph.verify_output()?;
    }
    Ok(())
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

//! Checked F32/F64 `ReLU` through finite dispatch and the exact HIP source emitted from that IR.

#[path = "alloc.rs"]
#[allow(dead_code)]
mod alloc;

#[rustfmt::skip]
use std::{
    error::Error,
    ffi::CStr,
    hint::black_box,
    mem::size_of,
    time::{Duration, Instant},
};
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuInvocationShape,
    PcuOwnedBinding,
    PcuOwnedCompletion,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    compile_hip_source,
    lower_dispatch_to_hip_source,
};

const BLOCK: u32 = 256;
const EXTENTS: [usize; 2] = [65, 1_048_576];
trait FloatCase: bytemuck::Pod + Copy + Send + Sync + std::fmt::Debug + 'static {
    type Bits: bytemuck::Pod + Copy + Default + Eq + std::fmt::Debug;
    const SCALAR: PcuScalarType;
    const CAPS: PcuValueTypeCaps;
    const LABEL: &'static str;
    const NATIVE_NAME: &'static CStr;
    fn bits(self) -> Self::Bits;
    fn from_bits_word(word: u64) -> Self;
    fn invalid() -> Self;
    fn value(index: usize, sample: u32) -> Self;
    fn expected(value: Self, reject_subnormal: bool) -> (Self, Option<u64>);
}

impl FloatCase for f32 {
    type Bits = u32;
    const SCALAR: PcuScalarType = PcuScalarType::F32;
    const CAPS: PcuValueTypeCaps = PcuValueTypeCaps::FLOAT32;
    const LABEL: &'static str = "f32";
    const NATIVE_NAME: &'static CStr = c"checked_relu_f32";
    fn bits(self) -> Self::Bits {
        self.to_bits()
    }
    fn from_bits_word(word: u64) -> Self {
        Self::from_bits(u32::try_from(word).expect("f32 bit pattern fits u32"))
    }
    fn invalid() -> Self {
        Self::INFINITY
    }
    fn value(index: usize, sample: u32) -> Self {
        match (index + usize::try_from(sample % 4).expect("bounded sample")) % 6 {
            0 => -3.25,
            1 => 0.0,
            2 => -0.0,
            3 => 2.5,
            4 => Self::from_bits(1),
            _ => -Self::from_bits(1),
        }
    }
    fn expected(value: Self, reject_subnormal: bool) -> (Self, Option<u64>) {
        let bits = value.to_bits();
        if reject_subnormal && bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0 && bits >> 31 == 0
        {
            (value, Some(4))
        } else if bits >> 31 != 0 {
            (0.0, None)
        } else {
            (value, None)
        }
    }
}

impl FloatCase for f64 {
    type Bits = u64;
    const SCALAR: PcuScalarType = PcuScalarType::F64;
    const CAPS: PcuValueTypeCaps = PcuValueTypeCaps::FLOAT64;
    const LABEL: &'static str = "f64";
    const NATIVE_NAME: &'static CStr = c"checked_relu_f64";
    fn bits(self) -> Self::Bits {
        self.to_bits()
    }
    fn from_bits_word(word: u64) -> Self {
        Self::from_bits(word)
    }
    fn invalid() -> Self {
        Self::INFINITY
    }
    fn value(index: usize, sample: u32) -> Self {
        match (index + usize::try_from(sample % 4).expect("bounded sample")) % 6 {
            0 => -3.25,
            1 => 0.0,
            2 => -0.0,
            3 => 2.5,
            4 => Self::from_bits(1),
            _ => -Self::from_bits(1),
        }
    }
    fn expected(value: Self, reject_subnormal: bool) -> (Self, Option<u64>) {
        let bits = value.to_bits();
        if reject_subnormal
            && bits & 0x7ff0_0000_0000_0000 == 0
            && bits & 0x000f_ffff_ffff_ffff != 0
            && bits >> 63 == 0
        {
            (value, Some(4))
        } else if bits >> 63 != 0 {
            (0.0, None)
        } else {
            (value, None)
        }
    }
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = crate::support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect::<Vec<_>>();
    let (backend, selected) =
        crate::support::selection::open_ranked(&discovery, candidates, BLOCK)?;
    let architecture = selected
        .architecture
        .as_deref()
        .ok_or("missing architecture")?;
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        device: Some(selected.device.id),
        ..fusion_pcu::global::PcuExecutionPolicy::default()
    })?;
    println!(
        "Checked F32/F64 ReLU on {} ({architecture}): native HIP is generated from the identical checked dispatch IR; resident times omit refresh transfers; full-host times include upload, dispatch/completion, and output readback.",
        selected.name
    );
    let runtime = discovery.open_device(selected.device)?;
    run_type::<f32>(criterion, &backend, &runtime, architecture)?;
    run_type::<f64>(criterion, &backend, &runtime, architecture)?;
    Ok(())
}

fn run_type<T: FloatCase>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    architecture: &str,
) -> Result<(), Box<dyn Error>> {
    for elements in EXTENTS {
        run_case::<T>(criterion, backend, runtime, architecture, elements, false)?;
        run_case::<T>(criterion, backend, runtime, architecture, elements, true)?;
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)] // Criterion owns the group through all registered cases.
#[allow(unsafe_code)] // The native route launches the HIP source emitted from this same PCU IR.
fn run_case<T: FloatCase>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    architecture: &str,
    elements: usize,
    recover_subnormal: bool,
) -> Result<(), Box<dyn Error>> {
    let count = u32::try_from(elements)?;
    let mode = if recover_subnormal {
        "clamp-recover"
    } else {
        "nominal"
    };
    let underflow = if recover_subnormal {
        PcuFloatUnderflowPolicy::RejectSubnormalResult
    } else {
        PcuFloatUnderflowPolicy::IeeeAfterRounding
    };
    let range = if recover_subnormal {
        fusion_pcu::PcuRangePolicy::Clamp
    } else {
        fusion_pcu::PcuRangePolicy::Reject
    };
    let value_type = PcuValueType::Scalar(T::SCALAR);
    let f32_bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        ),
    ];
    let f64_bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f64(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f64(),
        ),
    ];
    let typed_bindings = if T::SCALAR == PcuScalarType::F32 {
        &f32_bindings
    } else {
        &f64_bindings
    };
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type,
            op: PcuDispatchFloatUnaryOp::Relu,
            underflow_policy: underflow,
            range_policy: range,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: fusion_pcu::PcuKernelId(if T::SCALAR == PcuScalarType::F32 {
            0xf320
        } else {
            0xf640
        }),
        entry: PcuDispatchEntryPoint {
            name: "checked_relu",
            logical_shape: [count, 1, 1],
        },
        bindings: typed_bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: T::CAPS,
        feature_caps: PcuDispatchFeatureCaps::empty(),
    };
    let invocation_count = std::num::NonZeroU32::new(count).ok_or("empty extent")?;
    let prepared = crate::support::cold_once("checked ReLU dispatch preparation", || {
        backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(invocation_count),
        })
    })?;
    let generated = lower_dispatch_to_hip_source(&kernel)?;
    let native_source = generated.replace("fusion_kernel", T::NATIVE_NAME.to_str()?);
    let image = crate::support::cold_once("generated checked ReLU HIP compilation", || {
        compile_hip_source(&native_source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let native_kernel = module.function(T::NATIVE_NAME)?;
    let stream = runtime.create_stream()?;
    let byte_len = elements
        .checked_mul(size_of::<T>())
        .ok_or("byte length overflow")?;
    let grid = count.div_ceil(BLOCK);
    let mut input = vec![T::from_bits_word(0); elements];
    let mut output_bits = vec![T::Bits::default(); elements];
    fill::<T>(&mut input, 0);
    let mut pcu_input = backend.allocate(byte_len)?;
    let pcu_output = backend.allocate(byte_len)?;
    let mut native_input = runtime.allocate(byte_len)?;
    let native_output = runtime.allocate(byte_len)?;
    let mut native_fault = runtime.allocate(size_of::<u64>())?;
    pcu_input.copy_from(bytemuck::cast_slice(&input))?;
    native_input.copy_from(bytemuck::cast_slice(&input))?;
    let pcu_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(value_type),
            pcu_input.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(value_type),
            pcu_output.clone(),
        )?,
    ];
    let mut sequential = prepared.sequential_checked()?;
    verify_routes::<T>(
        &prepared,
        &mut sequential,
        &native_kernel,
        &stream,
        &pcu_bindings,
        &mut pcu_input,
        &pcu_output,
        &mut native_input,
        &native_output,
        &mut native_fault,
        &mut output_bits,
        &mut input,
        grid,
        recover_subnormal,
    )?;
    print_allocations::<T>(
        &prepared,
        &mut sequential,
        &native_kernel,
        &stream,
        &pcu_bindings,
        &native_input,
        &native_output,
        &mut native_fault,
        grid,
    )?;

    // These repeated diagnostics are actual paired samples, not Criterion confidence intervals.
    // Match the sequential factory's cold sentinel initialization outside both timers.
    {
        let mut diagnostic_sequential = prepared.sequential_checked()?;
        native_fault.copy_from(&u64::MAX.to_le_bytes())?;
        let mut fixture = PairedFixture::<T> {
            prepared: &prepared,
            sequential: &mut diagnostic_sequential,
            native_kernel: &native_kernel,
            stream: &stream,
            pcu_bindings: &pcu_bindings,
            pcu_input: &mut pcu_input,
            pcu_output: &pcu_output,
            native_input: &mut native_input,
            native_output: &native_output,
            native_fault: &mut native_fault,
            input: &mut input,
            output: &mut output_bits,
            grid,
            recover: recover_subnormal,
            native_sentinel_proven: true,
        };
        fixture.run(mode)?;
    }

    // Preflights/census may leave either a clean or faulted word. Establish this proof only from
    // each measured launch's terminal readback, exactly like the sequential PCU owner.
    let mut native_sentinel_proven = false;
    let mut group = criterion.benchmark_group(format!("checked-relu/{}/{}", T::LABEL, mode));
    group.throughput(Throughput::Elements(u64::from(count)));
    for resident in [true, false] {
        let label = if resident {
            "resident/dispatch+completion"
        } else {
            "full-host/upload+dispatch+readback"
        };
        group.bench_function(
            BenchmarkId::new("pcu", format!("{label}/{elements}")),
            |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for sample in 0..iterations {
                        fill::<T>(
                            &mut input,
                            u32::try_from(sample % 4).expect("bounded sample"),
                        );
                        let expected = expected_bits::<T>(&input, recover_subnormal);
                        let expected_fault = expected_fault_word::<T>(&input, recover_subnormal);
                        if resident {
                            pcu_input
                                .copy_from(bytemuck::cast_slice(&input))
                                .expect("PCU resident refresh");
                        }
                        let started = Instant::now();
                        if !resident {
                            pcu_input
                                .copy_from(bytemuck::cast_slice(&input))
                                .expect("PCU host upload");
                        }
                        let mut completion = prepared
                            .submit(&pcu_bindings)
                            .expect("PCU checked ReLU submit");
                        let outcome = completion.wait().expect("PCU checked ReLU completion");
                        if !resident {
                            pcu_output
                                .copy_to(bytemuck::cast_slice_mut(&mut output_bits))
                                .expect("PCU host readback");
                        }
                        elapsed += started.elapsed();
                        if resident {
                            pcu_output
                                .copy_to(bytemuck::cast_slice_mut(&mut output_bits))
                                .expect("PCU oracle readback");
                        }
                        assert_completion(outcome, recover_subnormal, expected_fault);
                        assert_eq!(output_bits, expected, "PCU checked ReLU bitwise oracle");
                        black_box(&expected);
                    }
                    elapsed
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(
                "pcu/sequential-reused-status",
                format!("{label}/{elements}"),
            ),
            |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for sample in 0..iterations {
                        fill::<T>(
                            &mut input,
                            u32::try_from(sample % 4).expect("bounded sample"),
                        );
                        let expected = expected_bits::<T>(&input, recover_subnormal);
                        let expected_fault = expected_fault_word::<T>(&input, recover_subnormal);
                        if resident {
                            pcu_input
                                .copy_from(bytemuck::cast_slice(&input))
                                .expect("PCU sequential resident refresh");
                        }
                        let started = Instant::now();
                        if !resident {
                            pcu_input
                                .copy_from(bytemuck::cast_slice(&input))
                                .expect("PCU sequential host upload");
                        }
                        let outcome = sequential
                            .submit_and_wait(&pcu_bindings)
                            .expect("PCU sequential checked ReLU completion");
                        if !resident {
                            pcu_output
                                .copy_to(bytemuck::cast_slice_mut(&mut output_bits))
                                .expect("PCU sequential host readback");
                        }
                        elapsed += started.elapsed();
                        if resident {
                            pcu_output
                                .copy_to(bytemuck::cast_slice_mut(&mut output_bits))
                                .expect("PCU sequential oracle readback");
                        }
                        assert_completion(outcome, recover_subnormal, expected_fault);
                        assert_eq!(
                            output_bits, expected,
                            "PCU sequential checked ReLU bitwise oracle"
                        );
                        black_box(&expected);
                    }
                    elapsed
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("native-generated-hip", format!("{label}/{elements}")),
            |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for sample in 0..iterations {
                        fill::<T>(
                            &mut input,
                            u32::try_from(sample % 4).expect("bounded sample"),
                        );
                        let expected = expected_bits::<T>(&input, recover_subnormal);
                        let expected_fault = expected_fault_word::<T>(&input, recover_subnormal);
                        if resident {
                            native_input
                                .copy_from(bytemuck::cast_slice(&input))
                                .expect("native resident refresh");
                        }
                        let started = Instant::now();
                        if !resident {
                            native_input
                                .copy_from(bytemuck::cast_slice(&input))
                                .expect("native host upload");
                        }
                        // Consume the prior proof before reset/launch; only terminal readback
                        // below may prove it clean again, including after recovered faults.
                        if !std::mem::take(&mut native_sentinel_proven) {
                            native_fault
                                .copy_from(&u64::MAX.to_le_bytes())
                                .expect("native fault reset");
                        }
                        {
                            let arguments = [
                                HipKernelArgument::Buffer(&native_input),
                                HipKernelArgument::Buffer(&native_output),
                                HipKernelArgument::Buffer(&native_fault),
                            ];
                            // SAFETY: Input/output cover `elements` T values and fault is one u64.
                            let mut completion = unsafe {
                                native_kernel.launch(
                                    &stream,
                                    [grid, 1, 1],
                                    [BLOCK, 1, 1],
                                    0,
                                    &arguments,
                                )
                            }
                            .expect("native checked ReLU launch");
                            completion.wait().expect("native checked ReLU completion");
                        }
                        let mut fault_bytes = [0_u8; 8];
                        native_fault
                            .copy_to(&mut fault_bytes)
                            .expect("native fault readback");
                        native_sentinel_proven = u64::from_le_bytes(fault_bytes) == u64::MAX;
                        if !resident {
                            native_output
                                .copy_to(bytemuck::cast_slice_mut(&mut output_bits))
                                .expect("native host readback");
                        }
                        elapsed += started.elapsed();
                        if resident {
                            native_output
                                .copy_to(bytemuck::cast_slice_mut(&mut output_bits))
                                .expect("native oracle readback");
                        }
                        assert_eq!(
                            u64::from_le_bytes(fault_bytes),
                            expected_fault,
                            "native checked ReLU fault oracle"
                        );
                        assert_eq!(output_bits, expected, "native checked ReLU bitwise oracle");
                        black_box(&expected);
                    }
                    elapsed
                });
            },
        );
    }
    group.finish();
    Ok(())
}

const PAIRED_TRIPLES: usize = 36;
// Each route occupies each position twice per six triples; repeat that balance six times.
const ROUTE_ORDERS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

// Only borrowed, already allocated resources enter the measured diagnostic paths.
struct PairedFixture<'a, 'dispatch, T: FloatCase> {
    prepared: &'a fusion_pcu_rocm::RocmPreparedDispatch,
    sequential: &'a mut fusion_pcu_rocm::RocmSequentialCheckedDispatch<'dispatch>,
    native_kernel: &'a HipKernel,
    stream: &'a fusion_pcu_rocm::HipStreamHandle,
    pcu_bindings: &'a [PcuOwnedBinding<DeviceBuffer>],
    pcu_input: &'a mut DeviceBuffer,
    pcu_output: &'a DeviceBuffer,
    native_input: &'a mut DeviceBuffer,
    native_output: &'a DeviceBuffer,
    native_fault: &'a mut DeviceBuffer,
    input: &'a mut [T],
    output: &'a mut [T::Bits],
    grid: u32,
    recover: bool,
    native_sentinel_proven: bool,
}

impl<T: FloatCase> PairedFixture<'_, '_, T> {
    fn run(&mut self, mode: &str) -> Result<(), Box<dyn Error>> {
        for resident in [true, false] {
            let mut seconds = [[0.0; PAIRED_TRIPLES]; 3];
            for triple in 0..PAIRED_TRIPLES {
                fill::<T>(self.input, u32::try_from(triple)?);
                let expected_fault = expected_fault_word::<T>(self.input, self.recover);
                for route in ROUTE_ORDERS[triple % ROUTE_ORDERS.len()] {
                    seconds[route][triple] = self.measure(route, resident, expected_fault)?;
                }
            }
            report_paired::<T>(&seconds, mode, self.input.len(), resident);
        }
        Ok(())
    }

    fn verify_output(&self) {
        for (index, (&actual, &input)) in self.output.iter().zip(self.input.iter()).enumerate() {
            assert_eq!(
                actual,
                T::expected(input, self.recover).0.bits(),
                "paired checked ReLU bitwise oracle at {index}"
            );
        }
    }

    fn measure(
        &mut self,
        route: usize,
        resident: bool,
        expected_fault: u64,
    ) -> Result<f64, Box<dyn Error>> {
        if resident {
            self.upload(route)?;
        }
        let started = Instant::now();
        if !resident {
            self.upload(route)?;
        }
        let outcome = match route {
            0 => {
                // Public submit intentionally owns an independent status per completion.
                let mut completion = self.prepared.submit(self.pcu_bindings)?;
                let outcome = completion.wait()?;
                if !resident {
                    self.read_output(route)?;
                }
                let elapsed = started.elapsed().as_secs_f64();
                // Completion destruction, reporting and all oracles are outside the timer.
                drop(completion);
                (elapsed, Some(outcome), None)
            }
            1 => {
                let outcome = self.sequential.submit_and_wait(self.pcu_bindings)?;
                if !resident {
                    self.read_output(route)?;
                }
                (started.elapsed().as_secs_f64(), Some(outcome), None)
            }
            2 => {
                let fault = self.launch_native()?;
                if !resident {
                    self.read_output(route)?;
                }
                (started.elapsed().as_secs_f64(), None, Some(fault))
            }
            _ => unreachable!("three diagnostic routes"),
        };
        if resident {
            self.read_output(route)?;
        }
        if let Some(completion) = outcome.1 {
            assert_completion(completion, self.recover, expected_fault);
        }
        if let Some(fault) = outcome.2 {
            assert_eq!(fault, expected_fault, "paired native fault oracle");
        }
        self.verify_output();
        Ok(outcome.0)
    }

    fn upload(&mut self, route: usize) -> Result<(), Box<dyn Error>> {
        let buffer = if route == 2 {
            &mut *self.native_input
        } else {
            &mut *self.pcu_input
        };
        buffer.copy_from(bytemuck::cast_slice(self.input))?;
        Ok(())
    }

    fn read_output(&mut self, route: usize) -> Result<(), Box<dyn Error>> {
        let buffer = if route == 2 {
            self.native_output
        } else {
            self.pcu_output
        };
        buffer.copy_to(bytemuck::cast_slice_mut(self.output))?;
        Ok(())
    }

    #[allow(unsafe_code)] // Native dispatch uses the same checked IR and buffer ABI.
    fn launch_native(&mut self) -> Result<u64, Box<dyn Error>> {
        // Match sequential PCU: an attempted launch consumes the clean terminal proof.
        if !std::mem::take(&mut self.native_sentinel_proven) {
            self.native_fault.copy_from(&u64::MAX.to_le_bytes())?;
        }
        {
            let arguments = [
                HipKernelArgument::Buffer(self.native_input),
                HipKernelArgument::Buffer(self.native_output),
                HipKernelArgument::Buffer(self.native_fault),
            ];
            // SAFETY: Both data buffers cover the T slices; status is exactly one u64.
            let mut completion = unsafe {
                self.native_kernel.launch(
                    self.stream,
                    [self.grid, 1, 1],
                    [BLOCK, 1, 1],
                    0,
                    &arguments,
                )
            }?;
            completion.wait()?;
        }
        let mut bytes = [0_u8; 8];
        self.native_fault.copy_to(&mut bytes)?;
        let fault = u64::from_le_bytes(bytes);
        self.native_sentinel_proven = fault == u64::MAX;
        Ok(fault)
    }
}

fn paired_median(mut values: [f64; PAIRED_TRIPLES]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    f64::midpoint(values[PAIRED_TRIPLES / 2 - 1], values[PAIRED_TRIPLES / 2])
}

fn report_paired<T: FloatCase>(
    seconds: &[[f64; PAIRED_TRIPLES]; 3],
    mode: &str,
    elements: usize,
    resident: bool,
) {
    let boundary = if resident { "resident" } else { "full-host" };
    let mut public_ratios = [0.0; PAIRED_TRIPLES];
    let mut sequential_ratios = [0.0; PAIRED_TRIPLES];
    for triple in 0..PAIRED_TRIPLES {
        public_ratios[triple] = seconds[0][triple] / seconds[2][triple];
        sequential_ratios[triple] = seconds[1][triple] / seconds[2][triple];
        println!(
            "Paired checked ReLU {} {mode}/{elements}/{boundary} triple={triple:02} order={:?} public_us={:.3} sequential_us={:.3} native_us={:.3} public/native={:.4} sequential/native={:.4}",
            T::LABEL,
            ROUTE_ORDERS[triple % ROUTE_ORDERS.len()],
            seconds[0][triple] * 1e6,
            seconds[1][triple] * 1e6,
            seconds[2][triple] * 1e6,
            public_ratios[triple],
            sequential_ratios[triple],
        );
    }
    // Median actual per-triple ratios, never a ratio of separately computed medians.
    println!(
        "Paired checked ReLU {} {mode}/{elements}/{boundary}: 36 interleaved triples, all six orders repeated; median_us public={:.3} sequential={:.3} native={:.3}; median_actual_ratio public/native={:.4} sequential/native={:.4}. Repeated diagnostics only; no paired CI or speedup inference. Routes 0=public independent-status, 1=sequential reused-status, 2=native generated HIP.",
        T::LABEL,
        paired_median(seconds[0]) * 1e6,
        paired_median(seconds[1]) * 1e6,
        paired_median(seconds[2]) * 1e6,
        paired_median(public_ratios),
        paired_median(sequential_ratios),
    );
}

fn fill<T: FloatCase>(values: &mut [T], sample: u32) {
    for (index, value) in values.iter_mut().enumerate() {
        *value = T::value(index, sample);
    }
}

fn expected_bits<T: FloatCase>(values: &[T], recover: bool) -> Vec<T::Bits> {
    values
        .iter()
        .map(|value| T::expected(*value, recover).0.bits())
        .collect()
}

fn expected_fault_word<T: FloatCase>(values: &[T], recover: bool) -> u64 {
    if !recover {
        return u64::MAX;
    }
    let index = values
        .iter()
        .position(|value| T::expected(*value, true).1.is_some())
        .expect("clamp-recovery workload contains a positive subnormal");
    0x8000_0000_0000_0000 | (u64::try_from(index).expect("extent fits fault word") << 3) | 4
}

fn assert_completion(outcome: PcuCompletionOutcome, recover: bool, expected_fault: u64) {
    if recover {
        let expected_index = (expected_fault >> 3) & 0x0fff_ffff_ffff_ffff;
        assert!(
            matches!(outcome, PcuCompletionOutcome::Fault(fault) if fault.recovered && fault.kind == fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow && fault.invocation_id == expected_index)
        );
    } else {
        assert_eq!(outcome, PcuCompletionOutcome::Succeeded);
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines, unsafe_code)]
// Keep the three routes and their fault/retry preflights together for comparison.
fn verify_routes<T: FloatCase>(
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
    sequential: &mut fusion_pcu_rocm::RocmSequentialCheckedDispatch<'_>,
    native_kernel: &HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    pcu_bindings: &[PcuOwnedBinding<DeviceBuffer>],
    pcu_input: &mut DeviceBuffer,
    pcu_output: &DeviceBuffer,
    native_input: &mut DeviceBuffer,
    native_output: &DeviceBuffer,
    native_fault: &mut DeviceBuffer,
    output: &mut [T::Bits],
    input: &mut [T],
    grid: u32,
    recover: bool,
) -> Result<(), Box<dyn Error>> {
    let expected = expected_bits::<T>(input, recover);
    let expected_fault = expected_fault_word::<T>(input, recover);
    let mut completion = prepared.submit(pcu_bindings)?;
    assert_completion(completion.wait()?, recover, expected_fault);
    pcu_output.copy_to(bytemuck::cast_slice_mut(output))?;
    assert_eq!(output, expected, "PCU checked ReLU bitwise oracle");
    let outcome = sequential.submit_and_wait(pcu_bindings)?;
    assert_completion(outcome, recover, expected_fault);
    pcu_output.copy_to(bytemuck::cast_slice_mut(output))?;
    assert_eq!(
        output, expected,
        "PCU sequential checked ReLU bitwise oracle"
    );
    native_fault.copy_from(&u64::MAX.to_le_bytes())?;
    {
        let arguments = [
            HipKernelArgument::Buffer(native_input),
            HipKernelArgument::Buffer(native_output),
            HipKernelArgument::Buffer(native_fault),
        ];
        // SAFETY: Input/output span the supplied slices and fault is exactly one u64.
        let mut native_completion =
            unsafe { native_kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments) }?;
        native_completion.wait()?;
    }
    let mut fault_bytes = [0_u8; 8];
    native_fault.copy_to(&mut fault_bytes)?;
    assert_eq!(
        u64::from_le_bytes(fault_bytes),
        expected_fault,
        "generated HIP ReLU fault oracle"
    );
    native_output.copy_to(bytemuck::cast_slice_mut(output))?;
    assert_eq!(
        output, expected,
        "generated HIP checked ReLU bitwise oracle"
    );
    input[0] = T::invalid();
    pcu_input.copy_from(bytemuck::cast_slice(input))?;
    native_input.copy_from(bytemuck::cast_slice(input))?;
    let mut fatal = prepared.submit(pcu_bindings)?;
    assert!(
        matches!(fatal.wait()?, PcuCompletionOutcome::Fault(fault) if !fault.recovered && fault.kind == fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand && fault.invocation_id == 0)
    );
    let sequential_fatal = sequential.submit_and_wait(pcu_bindings)?;
    assert!(
        matches!(sequential_fatal, PcuCompletionOutcome::Fault(fault) if !fault.recovered && fault.kind == fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand && fault.invocation_id == 0)
    );
    native_fault.copy_from(&u64::MAX.to_le_bytes())?;
    {
        let fatal_arguments = [
            HipKernelArgument::Buffer(native_input),
            HipKernelArgument::Buffer(native_output),
            HipKernelArgument::Buffer(native_fault),
        ];
        // SAFETY: Fatal preflight uses the same valid allocations and kernel ABI.
        let mut fatal_native = unsafe {
            native_kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &fatal_arguments)
        }?;
        fatal_native.wait()?;
    }
    let mut fault_bytes = [0_u8; 8];
    native_fault.copy_to(&mut fault_bytes)?;
    assert_eq!(
        u64::from_le_bytes(fault_bytes),
        5,
        "native fatal ReLU fault oracle"
    );
    input.fill(T::from_bits_word(0));
    pcu_input.copy_from(bytemuck::cast_slice(input))?;
    native_input.copy_from(bytemuck::cast_slice(input))?;
    let mut retry = prepared.submit(pcu_bindings)?;
    assert_eq!(
        retry.wait()?,
        PcuCompletionOutcome::Succeeded,
        "public status succeeds after terminal faults"
    );
    pcu_output.copy_to(bytemuck::cast_slice_mut(output))?;
    assert_eq!(output, expected_bits::<T>(input, false));
    assert_eq!(
        sequential.submit_and_wait(pcu_bindings)?,
        PcuCompletionOutcome::Succeeded,
        "sequential status resets and succeeds after terminal faults"
    );
    pcu_output.copy_to(bytemuck::cast_slice_mut(output))?;
    assert_eq!(output, expected_bits::<T>(input, false));
    native_fault.copy_from(&u64::MAX.to_le_bytes())?;
    {
        let retry_arguments = [
            HipKernelArgument::Buffer(native_input),
            HipKernelArgument::Buffer(native_output),
            HipKernelArgument::Buffer(native_fault),
        ];
        // SAFETY: Retry uses the same valid allocations and kernel ABI.
        let mut retry_native = unsafe {
            native_kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &retry_arguments)
        }?;
        retry_native.wait()?;
    }
    native_fault.copy_to(&mut fault_bytes)?;
    assert_eq!(u64::from_le_bytes(fault_bytes), u64::MAX);
    native_output.copy_to(bytemuck::cast_slice_mut(output))?;
    assert_eq!(output, expected_bits::<T>(input, false));
    fill::<T>(input, 0);
    pcu_input.copy_from(bytemuck::cast_slice(input))?;
    native_input.copy_from(bytemuck::cast_slice(input))?;
    Ok(())
}

#[allow(clippy::too_many_arguments, unsafe_code)]
fn print_allocations<T: FloatCase>(
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
    sequential: &mut fusion_pcu_rocm::RocmSequentialCheckedDispatch<'_>,
    native_kernel: &HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    pcu_bindings: &[PcuOwnedBinding<DeviceBuffer>],
    native_input: &DeviceBuffer,
    native_output: &DeviceBuffer,
    native_fault: &mut DeviceBuffer,
    grid: u32,
) -> Result<(), Box<dyn Error>> {
    let capture = alloc::AllocationCapture::start();
    let mut completion = prepared.submit(pcu_bindings)?;
    let _ = completion.wait()?;
    drop(completion);
    let pcu = alloc::AllocationCapture::finish();
    drop(capture);
    let capture = alloc::AllocationCapture::start();
    let _ = sequential.submit_and_wait(pcu_bindings)?;
    let sequential_counts = alloc::AllocationCapture::finish();
    drop(capture);
    let capture = alloc::AllocationCapture::start();
    native_fault.copy_from(&u64::MAX.to_le_bytes())?;
    let args = [
        HipKernelArgument::Buffer(native_input),
        HipKernelArgument::Buffer(native_output),
        HipKernelArgument::Buffer(native_fault),
    ];
    // SAFETY: Input/output are sized to `input`; the status owner is one u64.
    let mut completion =
        unsafe { native_kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
    completion.wait()?;
    drop(completion);
    let mut fault_bytes = [0_u8; 8];
    native_fault.copy_to(&mut fault_bytes)?;
    black_box(u64::from_le_bytes(fault_bytes));
    let native = alloc::AllocationCapture::finish();
    drop(capture);
    println!(
        "Checked {} ReLU resident host allocation census per call (alloc/realloc/dealloc/requested bytes; excludes HIP/driver and resident buffers). Public submit has per-completion status ownership; sequential route reuses its preallocated status word on success and resets after a fault; HIP owns a separate status buffer. Public PCU {}/{}/{}/{}; sequential PCU {}/{}/{}/{}; generated HIP {}/{}/{}/{}.",
        T::LABEL,
        pcu.alloc_calls,
        pcu.realloc_calls,
        pcu.dealloc_calls,
        pcu.requested_bytes,
        sequential_counts.alloc_calls,
        sequential_counts.realloc_calls,
        sequential_counts.dealloc_calls,
        sequential_counts.requested_bytes,
        native.alloc_calls,
        native.realloc_calls,
        native.dealloc_calls,
        native.requested_bytes
    );
    Ok(())
}

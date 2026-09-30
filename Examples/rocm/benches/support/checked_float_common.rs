//! Shared checked floating-point benchmark setup, measurements and paired diagnostics.
#[path = "alloc.rs"]
#[allow(dead_code)]
pub(super) mod alloc;

// Emit concrete scalar signatures for the frontend while sharing every measurement boundary.
// This keeps width selection out of timed execution and owns exactly one census allocator.
macro_rules! define_checked_float_benchmark {
    ($scalar:ident, $bits:ident, $caps:ident, $group:literal, $kernel_id:literal, $half_ulp:literal, $min_normal:literal) => {
        define_checked_float_benchmark!($scalar, $bits, $caps, $group, $kernel_id, $half_ulp, $min_normal, Add, +);
    };
    ($scalar:ident, $bits:ident, $caps:ident, $group:literal, $kernel_id:literal, $half_ulp:literal, $min_normal:literal, $operation:ident, $operator:tt) => {
        use super::common::alloc;
        #[rustfmt::skip]
        use std::{
            error::Error,
            mem::size_of,
            num::NonZeroU32,
            time::Duration,
            time::Instant,
        };

        #[rustfmt::skip]
        use criterion::{
            BenchmarkId,
            Criterion,
            Throughput,
        };
        #[allow(unused_imports)] // The PCU macro consumes these paths before ordinary Rust name resolution.
        use fusion_pcu::pcu;
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
            PcuDispatchFloatBinaryOp,
            PcuDispatchIndex,
            PcuDispatchKernelIr,
            PcuDispatchOp,
            PcuDispatchSubmission,
            PcuDispatchValueId,
            PcuExecutionFault,
            PcuExecutionFaultKind,
            PcuFloatUnderflowPolicy,
            PcuInvocationShape,
            PcuOwnedCompletion,
            PcuValueType,
            PcuValueTypeCaps,
            PcuExecutionError,
            PcuTensor,
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
        const POLICY: PcuFloatUnderflowPolicy = PcuFloatUnderflowPolicy::IeeeAfterRounding;
        const BINDINGS: &[PcuBinding<'static>] = &[
            PcuBinding::value(
                Some("lhs"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::$scalar(),
            ),
            PcuBinding::value(
                Some("rhs"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::$scalar(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::$scalar(),
            ),
        ];
        const OPS: &[PcuDispatchOp<'static>] = &[
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                value_type: PcuValueType::$scalar(),
                op: PcuDispatchFloatBinaryOp::$operation,
                underflow_policy: POLICY,
                range_policy: fusion_pcu::PcuRangePolicy::Reject,
                result: PcuDispatchValueId(3),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];

        #[fusion_pcu::pcu]
        fn source_binary(lhs: &[$scalar], rhs: &[$scalar]) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
            Ok(lhs $operator rhs)
        }

        #[fusion_pcu::pcu]
        fn source_identity(input: &[$scalar]) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
            pcu::identity(input)
        }

        #[fusion_pcu::pcu(invocations = N)]
        fn source_refresh<const N: usize>(input: &[$scalar], output: &mut [$scalar]) {
            let id = pcu::context::global_invocation_id();
            output[id] = input[id];
        }

        fn kernel(elements: usize) -> Result<PcuDispatchKernelIr<'static>, Box<dyn Error>> {
            Ok(PcuDispatchKernelIr {
                id: fusion_pcu::PcuKernelId($kernel_id),
                entry: PcuDispatchEntryPoint {
                    name: $group,
                    logical_shape: [u32::try_from(elements)?, 1, 1],
                },
                bindings: BINDINGS,
                ports: &[],
                parameters: &[],
                ops: OPS,
                type_caps: PcuValueTypeCaps::$caps,
                feature_caps: PcuDispatchFeatureCaps::empty(),
            })
        }

        pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
            let discovery = RocmDiscovery::new();
            let candidates = crate::support::selected_candidates(&discovery)?
                .into_iter()
                .filter(|candidate| candidate.architecture.is_some())
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                return Err(
                    "no selected ROCm device has an architecture for native HIP compilation".into(),
                );
            }
            let (backend, selected) =
                crate::support::selection::open_ranked(&discovery, candidates, BLOCK)?;
            fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
                backend: fusion_pcu::global::PcuBackendChoice::Rocm,
                device: Some(selected.device.id),
                ..fusion_pcu::global::PcuExecutionPolicy::default()
            })?;
            let architecture = selected
                .architecture
                .as_deref()
                .ok_or("missing architecture")?;
            println!(
                "Checked {} binary benchmark device: {} ({architecture}); default IEEE underflow policy preserves exact subnormals. Native HIP compiles the same generated helper as PCU, so this compares dispatch/completion overhead rather than arithmetic implementations.",
                stringify!($scalar), selected.name
            );
            for elements in [65, 1_048_576] {
                let runtime = discovery.open_device(selected.device)?;
                run_case(criterion, &backend, &runtime, architecture, elements)?;
            }
            Ok(())
        }

        #[allow(clippy::too_many_lines)]
        #[allow(clippy::significant_drop_tightening)]
        #[allow(unsafe_code)] // HIP launch calls are required for the native route in this benchmark.
        fn run_case(
            criterion: &mut Criterion,
            backend: &RocmOwnedDispatchBackend,
            runtime: &HipRuntime,
            architecture: &str,
            elements: usize,
        ) -> Result<(), Box<dyn Error>> {
            let logical_count = u32::try_from(elements)?;
            let invocation_count = NonZeroU32::new(logical_count).ok_or("zero benchmark extent")?;
            let kernel = kernel(elements)?;
            let prepared = crate::support::cold_once("checked float PCU preparation", || {
                backend.prepare_dispatch(PcuDispatchSubmission {
                    kernel: &kernel,
                    shape: PcuInvocationShape::invocations(invocation_count),
                })
            })?;
            let hip_source = lower_dispatch_to_hip_source(&kernel)?;
            let native_source = hip_source.replace("fusion_kernel", "native_checked_float_binary");
            let image = crate::support::cold_once("checked float native HIP compilation", || {
                compile_hip_source(&native_source, architecture)
            })?;
            let native_module = runtime.load_module(&image)?;
            let native_kernel = native_module.function(c"native_checked_float_binary")?;
            let native_stream = runtime.create_stream()?;
            let byte_len = elements
                .checked_mul(size_of::<$scalar>())
                .ok_or("byte extent overflow")?;
            let (mut left, mut right) = benchmark_inputs(elements, 0);
            let mut source_left = source_identity(&left)?;
            let mut source_right = source_identity(&right)?;
            verify_source_fault(
                source_binary(&[1.0, $scalar::INFINITY], &[1.0, 0.0]),
                PcuExecutionFaultKind::InvalidFloatingOperand,
            );
            verify_source_fault(
                source_binary(&[1.0, $scalar::MAX], &[1.0, overflow_rhs()]),
                PcuExecutionFaultKind::ArithmeticOverflow,
            );
            if matches!(PcuDispatchFloatBinaryOp::$operation, PcuDispatchFloatBinaryOp::Div) {
                verify_source_fault(
                    source_binary(&[1.0, 0.0], &[1.0, 0.0]),
                    PcuExecutionFaultKind::DivideByZero,
                );
            }
            let mut pcu_left = backend.allocate(byte_len)?;
            let mut pcu_right = backend.allocate(byte_len)?;
            pcu_left.copy_from(bytemuck::cast_slice(&left))?;
            pcu_right.copy_from(bytemuck::cast_slice(&right))?;
            let mut native_left = runtime.allocate(byte_len)?;
            let mut native_right = runtime.allocate(byte_len)?;
            native_left.copy_from(bytemuck::cast_slice(&left))?;
            native_right.copy_from(bytemuck::cast_slice(&right))?;
            let grid = logical_count.div_ceil(BLOCK);
            preflight_faults(
                backend,
                runtime,
                &prepared,
                &native_kernel,
                &native_stream,
                &mut pcu_left,
                &mut pcu_right,
                &mut native_left,
                &mut native_right,
                elements,
                grid,
            )?;
            pcu_left.copy_from(bytemuck::cast_slice(&left))?;
            pcu_right.copy_from(bytemuck::cast_slice(&right))?;
            native_left.copy_from(bytemuck::cast_slice(&left))?;
            native_right.copy_from(bytemuck::cast_slice(&right))?;
            let expected = expected_bits(&left, &right);
            let mut observed_bits = vec![<$bits>::default(); elements];
            let mut source_output = vec![0.0; elements];
            let source_result = source_binary(&source_left, &source_right)?;
            source_result.read_into(&mut source_output)?;
            assert_eq!(
                source_output
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected,
                "#[pcu] source checked-float output oracle"
            );
            let pcu_output = backend.allocate(byte_len)?;
            let pcu_bindings = [
                backend.binding(
                    PcuBindingRef::new(0, 0),
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::$scalar()),
                    pcu_left.clone(),
                )?,
                backend.binding(
                    PcuBindingRef::new(0, 1),
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::$scalar()),
                    pcu_right.clone(),
                )?,
                backend.binding(
                    PcuBindingRef::new(0, 2),
                    PcuBindingAccess::WriteOnly,
                    PcuBindingType::Value(PcuValueType::$scalar()),
                    pcu_output.clone(),
                )?,
            ];
            let mut pcu_completion = prepared.submit(&pcu_bindings)?;
            assert_eq!(pcu_completion.wait()?, PcuCompletionOutcome::Succeeded);
            drop(pcu_completion);
            pcu_output.copy_to(bytemuck::cast_slice_mut(&mut observed_bits))?;
            assert_eq!(observed_bits, expected, "PCU checked-float output oracle");
            drop((pcu_output, pcu_bindings));

            let native_output = runtime.allocate(byte_len)?;
            let mut native_fault = runtime.allocate(size_of::<u64>())?;
            native_fault.copy_from(&u64::MAX.to_le_bytes())?;
            let args = [
                HipKernelArgument::Buffer(&native_left),
                HipKernelArgument::Buffer(&native_right),
                HipKernelArgument::Buffer(&native_output),
                HipKernelArgument::Buffer(&native_fault),
            ];
            // SAFETY: Each resident allocation spans exactly `elements` $scalar values; fault is 8 bytes.
            let mut native_completion =
                unsafe { native_kernel.launch(&native_stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
            native_completion.wait()?;
            drop(native_completion);
            let mut native_fault_bytes = [0_u8; 8];
            native_fault.copy_to(&mut native_fault_bytes)?;
            assert_eq!(u64::from_le_bytes(native_fault_bytes), u64::MAX);
            native_output.copy_to(bytemuck::cast_slice_mut(&mut observed_bits))?;
            assert_eq!(
                observed_bits, expected,
                "native checked-float output oracle"
            );
            drop((native_output, native_fault));

            assert_allocator_census_is_live();
            let (source_allocs, pcu_allocs, native_allocs) = allocation_census(
                &left,
                &right,
                &source_left,
                &source_right,
                &mut source_output,
                &mut observed_bits,
                backend,
                runtime,
                &prepared,
                &native_kernel,
                &native_stream,
                &pcu_left,
                &pcu_right,
                &native_left,
                &native_right,
                byte_len,
                grid,
            )?;
            println!(
                "Checked {} binary {elements} Rust heap census (alloc/realloc/dealloc/requested bytes; excludes HIP/driver): source {}/{}/{}/{}, PCU {}/{}/{}/{}, native {}/{}/{}/{}.",
                stringify!($scalar),
                source_allocs.alloc_calls,
                source_allocs.realloc_calls,
                source_allocs.dealloc_calls,
                source_allocs.requested_bytes,
                pcu_allocs.alloc_calls,
                pcu_allocs.realloc_calls,
                pcu_allocs.dealloc_calls,
                pcu_allocs.requested_bytes,
                native_allocs.alloc_calls,
                native_allocs.realloc_calls,
                native_allocs.dealloc_calls,
                native_allocs.requested_bytes,
            );

            // Resolve the resident refresh kernel before Criterion calibration; compilation
            // is excluded from the measured call but must not distort iteration calibration.
            refresh_benchmark_inputs(
                elements, 0, &mut left, &mut right, &mut source_left, &mut source_right,
                &mut pcu_left, &mut pcu_right, &mut native_left, &mut native_right,
            );

            let mut group = criterion.benchmark_group($group);
            group.throughput(Throughput::Elements(u64::from(logical_count)));
            group.bench_function(BenchmarkId::new("pcu-source/checked-binary", elements), |b| {
                b.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    let mut sample = 1_u32;
                    for _ in 0..iterations {
                        fill_inputs(&mut left, &mut right, sample);
                        sample = sample.wrapping_add(1);
                        match elements {
                            65 => {
                                source_refresh::<65>(&left, &mut source_left)
                                    .expect("source lhs refresh");
                                source_refresh::<65>(&right, &mut source_right)
                                    .expect("source rhs refresh");
                            }
                            1_048_576 => {
                                source_refresh::<1_048_576>(&left, &mut source_left)
                                    .expect("source lhs refresh");
                                source_refresh::<1_048_576>(&right, &mut source_right)
                                    .expect("source rhs refresh");
                            }
                            _ => unreachable!("benchmark has fixed extents"),
                        }
                        total += measure_source(
                            &source_left,
                            &source_right,
                            &mut source_output,
                            &left,
                            &right,
                        );
                    }
                    total
                });
            });
            group.bench_function(BenchmarkId::new("pcu/same-helper", elements), |b| {
                b.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    let mut sample = 1_u32;
                    for _ in 0..iterations {
                        fill_inputs(&mut left, &mut right, sample);
                        sample = sample.wrapping_add(1);
                        pcu_left
                            .copy_from(bytemuck::cast_slice(&left))
                            .expect("PCU lhs refresh");
                        pcu_right
                            .copy_from(bytemuck::cast_slice(&right))
                            .expect("PCU rhs refresh");
                        total += measure_pcu(
                            backend,
                            &prepared,
                            &pcu_left,
                            &pcu_right,
                            byte_len,
                            &left,
                            &right,
                            &mut observed_bits,
                        );
                    }
                    total
                });
            });
            group.bench_function(BenchmarkId::new("native-hip/same-helper", elements), |b| {
                b.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    let mut sample = 1_u32;
                    for _ in 0..iterations {
                        fill_inputs(&mut left, &mut right, sample);
                        sample = sample.wrapping_add(1);
                        native_left
                            .copy_from(bytemuck::cast_slice(&left))
                            .expect("native lhs refresh");
                        native_right
                            .copy_from(bytemuck::cast_slice(&right))
                            .expect("native rhs refresh");
                        total += measure_native(
                            runtime,
                            &native_kernel,
                            &native_stream,
                            &native_left,
                            &native_right,
                            byte_len,
                            grid,
                            &left,
                            &right,
                            &mut observed_bits,
                        );
                    }
                    total
                });
            });
            group.finish();
            run_paired_diagnostic(
                elements,
                &mut left,
                &mut right,
                &mut source_left,
                &mut source_right,
                &mut source_output,
                &mut observed_bits,
                backend,
                runtime,
                &prepared,
                &native_kernel,
                &native_stream,
                &mut pcu_left,
                &mut pcu_right,
                &mut native_left,
                &mut native_right,
                byte_len,
                grid,
            );
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn run_paired_diagnostic(
            elements: usize,
            left: &mut [$scalar],
            right: &mut [$scalar],
            source_left: &mut PcuTensor<$scalar>,
            source_right: &mut PcuTensor<$scalar>,
            source_output: &mut [$scalar],
            observed_bits: &mut [$bits],
            backend: &RocmOwnedDispatchBackend,
            runtime: &HipRuntime,
            prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
            native_kernel: &HipKernel,
            native_stream: &fusion_pcu_rocm::HipStreamHandle,
            pcu_left: &mut DeviceBuffer,
            pcu_right: &mut DeviceBuffer,
            native_left: &mut DeviceBuffer,
            native_right: &mut DeviceBuffer,
            byte_len: usize,
            grid: u32,
        ) {
            let routes = [
                [0usize, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ];
            let mut durations: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(36));
            let mut source_raw = Vec::with_capacity(36);
            let mut source_native = Vec::with_capacity(36);
            let mut raw_native = Vec::with_capacity(36);
            for triple in 0..36u32 {
                refresh_benchmark_inputs(
                    elements,
                    triple.wrapping_add(1),
                    left,
                    right,
                    source_left,
                    source_right,
                    pcu_left,
                    pcu_right,
                    native_left,
                    native_right,
                );
                let mut paired = [Duration::ZERO; 3];
                for route in routes[usize::try_from(triple).expect("bounded triple index") % routes.len()] {
                    paired[route] = match route {
                        0 => measure_source(source_left, source_right, source_output, left, right),
                        1 => measure_pcu(
                            backend,
                            prepared,
                            pcu_left,
                            pcu_right,
                            byte_len,
                            left,
                            right,
                            observed_bits,
                        ),
                        2 => measure_native(
                            runtime,
                            native_kernel,
                            native_stream,
                            native_left,
                            native_right,
                            byte_len,
                            grid,
                            left,
                            right,
                            observed_bits,
                        ),
                        _ => unreachable!("paired route index is bounded"),
                    };
                }
                let nanos = paired.map(|duration| duration.as_secs_f64() * 1_000_000_000.0);
                for (route, duration) in nanos.into_iter().enumerate() {
                    durations[route].push(duration);
                }
                source_raw.push(nanos[0] / nanos[1]);
                source_native.push(nanos[0] / nanos[2]);
                raw_native.push(nanos[1] / nanos[2]);
            }
            println!(
                "Paired diagnostic {} {elements} (36 triples, all 6 route orders): median ns source/raw/native = {:.0}/{:.0}/{:.0}; paired median ratios source/raw={:.4}, source/native={:.4}, raw/native={:.4}",
                stringify!($scalar),
                median(&mut durations[0]),
                median(&mut durations[1]),
                median(&mut durations[2]),
                median(&mut source_raw),
                median(&mut source_native),
                median(&mut raw_native),
            );
        }

        #[allow(clippy::too_many_arguments)]
        fn refresh_benchmark_inputs(
            elements: usize,
            sample: u32,
            left: &mut [$scalar],
            right: &mut [$scalar],
            source_left: &mut PcuTensor<$scalar>,
            source_right: &mut PcuTensor<$scalar>,
            pcu_left: &mut DeviceBuffer,
            pcu_right: &mut DeviceBuffer,
            native_left: &mut DeviceBuffer,
            native_right: &mut DeviceBuffer,
        ) {
            fill_inputs(left, right, sample);
            match elements {
                65 => {
                    source_refresh::<65>(left, source_left).expect("source lhs refresh");
                    source_refresh::<65>(right, source_right).expect("source rhs refresh");
                }
                1_048_576 => {
                    source_refresh::<1_048_576>(left, source_left).expect("source lhs refresh");
                    source_refresh::<1_048_576>(right, source_right).expect("source rhs refresh");
                }
                _ => unreachable!("benchmark has fixed extents"),
            }
            pcu_left
                .copy_from(bytemuck::cast_slice(left))
                .expect("PCU lhs refresh");
            pcu_right
                .copy_from(bytemuck::cast_slice(right))
                .expect("PCU rhs refresh");
            native_left
                .copy_from(bytemuck::cast_slice(left))
                .expect("native lhs refresh");
            native_right
                .copy_from(bytemuck::cast_slice(right))
                .expect("native rhs refresh");
        }

        fn measure_source(
            source_left: &PcuTensor<$scalar>,
            source_right: &PcuTensor<$scalar>,
            output: &mut [$scalar],
            left: &[$scalar],
            right: &[$scalar],
        ) -> Duration {
            let start = Instant::now();
            let result = source_binary(source_left, source_right).expect("source checked float binary");
            let operation = start.elapsed();
            result
                .read_into(output)
                .expect("source checked float completion");
            verify_output(output, left, right);
            let release = Instant::now();
            drop(result);
            operation + release.elapsed()
        }

        #[allow(clippy::too_many_arguments)]
        fn measure_pcu(
            backend: &RocmOwnedDispatchBackend,
            prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
            pcu_left: &DeviceBuffer,
            pcu_right: &DeviceBuffer,
            byte_len: usize,
            left: &[$scalar],
            right: &[$scalar],
            observed: &mut [$bits],
        ) -> Duration {
            let start = Instant::now();
            let output = backend.allocate(byte_len).expect("PCU output allocation");
            let bindings = [
                backend
                    .binding(
                        PcuBindingRef::new(0, 0),
                        PcuBindingAccess::ReadOnly,
                        PcuBindingType::Value(PcuValueType::$scalar()),
                        pcu_left.clone(),
                    )
                    .expect("PCU left binding"),
                backend
                    .binding(
                        PcuBindingRef::new(0, 1),
                        PcuBindingAccess::ReadOnly,
                        PcuBindingType::Value(PcuValueType::$scalar()),
                        pcu_right.clone(),
                    )
                    .expect("PCU right binding"),
                backend
                    .binding(
                        PcuBindingRef::new(0, 2),
                        PcuBindingAccess::WriteOnly,
                        PcuBindingType::Value(PcuValueType::$scalar()),
                        output.clone(),
                    )
                    .expect("PCU output binding"),
            ];
            let mut completion = prepared.submit(&bindings).expect("PCU checked binary submit");
            assert_eq!(
                completion.wait().expect("PCU checked binary completion"),
                PcuCompletionOutcome::Succeeded
            );
            drop(completion);
            let operation = start.elapsed();
            output
                .copy_to(bytemuck::cast_slice_mut(observed))
                .expect("PCU checked binary output oracle");
            verify_bits(observed, left, right);
            let release = Instant::now();
            drop((output, bindings));
            operation + release.elapsed()
        }

        #[allow(unsafe_code, clippy::too_many_arguments)]
        fn measure_native(
            runtime: &HipRuntime,
            kernel: &HipKernel,
            stream: &fusion_pcu_rocm::HipStreamHandle,
            left_device: &DeviceBuffer,
            right_device: &DeviceBuffer,
            byte_len: usize,
            grid: u32,
            left: &[$scalar],
            right: &[$scalar],
            observed: &mut [$bits],
        ) -> Duration {
            let start = Instant::now();
            let output = runtime
                .allocate(byte_len)
                .expect("native output allocation");
            let mut fault = runtime
                .allocate(size_of::<u64>())
                .expect("native fault allocation");
            fault
                .copy_from(&u64::MAX.to_le_bytes())
                .expect("native fault initialization");
            let args = [
                HipKernelArgument::Buffer(left_device),
                HipKernelArgument::Buffer(right_device),
                HipKernelArgument::Buffer(&output),
                HipKernelArgument::Buffer(&fault),
            ];
            // SAFETY: Resident input/output buffers span the $scalar slices and fault is exactly 8 bytes.
            let mut completion = unsafe { kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }
                .expect("native checked binary launch");
            completion.wait().expect("native checked binary completion");
            drop(completion);
            let mut status = [0; 8];
            fault.copy_to(&mut status).expect("native fault readback");
            assert_eq!(u64::from_le_bytes(status), u64::MAX);
            let operation = start.elapsed();
            output
                .copy_to(bytemuck::cast_slice_mut(observed))
                .expect("native checked binary output oracle");
            verify_bits(observed, left, right);
            let release = Instant::now();
            drop((output, fault));
            operation + release.elapsed()
        }

        fn median(values: &mut [f64]) -> f64 {
            values.sort_by(f64::total_cmp);
            match values.len() {
                0 => 0.0,
                length if length % 2 == 0 => f64::midpoint(values[length / 2 - 1], values[length / 2]),
                length => values[length / 2],
            }
        }

        #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
        #[allow(unsafe_code)]
        fn allocation_census(
            left: &[$scalar],
            right: &[$scalar],
            source_left: &PcuTensor<$scalar>,
            source_right: &PcuTensor<$scalar>,
            source_output: &mut [$scalar],
            observed: &mut [$bits],
            backend: &RocmOwnedDispatchBackend,
            runtime: &HipRuntime,
            prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
            native_kernel: &HipKernel,
            native_stream: &fusion_pcu_rocm::HipStreamHandle,
            pcu_left: &DeviceBuffer,
            pcu_right: &DeviceBuffer,
            native_left: &DeviceBuffer,
            native_right: &DeviceBuffer,
            byte_len: usize,
            grid: u32,
        ) -> Result<
            (
                alloc::AllocationCounts,
                alloc::AllocationCounts,
                alloc::AllocationCounts,
            ),
            Box<dyn Error>,
        > {
            let source_capture = alloc::AllocationCapture::start();
            let source_output_tensor = source_binary(source_left, source_right)?;
            source_output_tensor.read_into(source_output)?;
            verify_output(source_output, left, right);
            drop(source_output_tensor);
            let source_counts = alloc::AllocationCapture::finish();
            drop(source_capture);

            let pcu_capture = alloc::AllocationCapture::start();
            let output = backend.allocate(byte_len)?;
            let bindings = [
                backend.binding(
                    PcuBindingRef::new(0, 0),
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::$scalar()),
                    pcu_left.clone(),
                )?,
                backend.binding(
                    PcuBindingRef::new(0, 1),
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::$scalar()),
                    pcu_right.clone(),
                )?,
                backend.binding(
                    PcuBindingRef::new(0, 2),
                    PcuBindingAccess::WriteOnly,
                    PcuBindingType::Value(PcuValueType::$scalar()),
                    output.clone(),
                )?,
            ];
            let mut completion = prepared.submit(&bindings)?;
            assert_eq!(completion.wait()?, PcuCompletionOutcome::Succeeded);
            drop(completion);
            output.copy_to(bytemuck::cast_slice_mut(observed))?;
            verify_bits(observed, left, right);
            drop((output, bindings));
            let pcu_counts = alloc::AllocationCapture::finish();
            drop(pcu_capture);

            let native_capture = alloc::AllocationCapture::start();
            let output = runtime.allocate(byte_len)?;
            let mut fault = runtime.allocate(8)?;
            fault.copy_from(&u64::MAX.to_le_bytes())?;
            let args = [
                HipKernelArgument::Buffer(native_left),
                HipKernelArgument::Buffer(native_right),
                HipKernelArgument::Buffer(&output),
                HipKernelArgument::Buffer(&fault),
            ];
            // SAFETY: Inputs/output span `observed.len()` floats; the fault allocation is 8 bytes.
            let mut completion =
                unsafe { native_kernel.launch(native_stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
            completion.wait()?;
            drop(completion);
            let mut status = [0; 8];
            fault.copy_to(&mut status)?;
            assert_eq!(u64::from_le_bytes(status), u64::MAX);
            output.copy_to(bytemuck::cast_slice_mut(observed))?;
            verify_bits(observed, left, right);
            drop((output, fault));
            let native_counts = alloc::AllocationCapture::finish();
            drop(native_capture);
            Ok((source_counts, pcu_counts, native_counts))
        }

        fn assert_allocator_census_is_live() {
            let capture = alloc::AllocationCapture::start();
            let probe = Box::new(0_u64);
            std::hint::black_box(&probe);
            drop(probe);
            let counts = alloc::AllocationCapture::finish();
            drop(capture);
            assert!(
                counts.alloc_calls > 0 && counts.dealloc_calls > 0,
                "Rust allocation census failed to observe a boxed probe"
            );
        }

        fn benchmark_inputs(elements: usize, sample: u32) -> (Vec<$scalar>, Vec<$scalar>) {
            let mut left = vec![0.0; elements];
            let mut right = vec![0.0; elements];
            fill_inputs(&mut left, &mut right, sample);
            (left, right)
        }

        const fn overflow_rhs() -> $scalar {
            if matches!(PcuDispatchFloatBinaryOp::$operation, PcuDispatchFloatBinaryOp::Div) {
                0.5
            } else {
                $scalar::MAX
            }
        }

        fn fill_inputs(left: &mut [$scalar], right: &mut [$scalar], sample: u32) {
            for (index, (lhs, rhs)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
                if matches!(PcuDispatchFloatBinaryOp::$operation, PcuDispatchFloatBinaryOp::Div) {
                    let bump = usize::try_from(sample % 97).expect("bounded sample");
                    *lhs = $scalar::from(u16::try_from((index + bump) % 10_000).expect("bounded numerator")) * 0.125;
                    *rhs = $scalar::from(u16::try_from((index + bump) % 97 + 1).expect("nonzero bounded denominator")) * 0.25;
                    continue;
                }
                match (index + usize::try_from(sample % 4).expect("small bounded sample")) % 4 {
                    0 => {
                        *lhs = $scalar::from_bits(1);
                        *rhs = $scalar::from_bits(1);
                    }
                    1 => {
                        *lhs = 1.0;
                        *rhs = $scalar::from_bits($half_ulp);
                    }
                    2 => {
                        *lhs = $scalar::from_bits($min_normal);
                        *rhs = -$scalar::from_bits(1);
                    }
                    _ => {
                        let bump = usize::try_from(sample % 8).expect("small bounded sample");
                        *lhs = $scalar::from(u16::try_from((index + bump) % 10_000).expect("bounded value"))
                            * 0.125;
                        *rhs = $scalar::from(u16::try_from((index + bump) % 97).expect("bounded value")) * 0.25;
                    }
                }
            }
        }

        fn expected_bits(left: &[$scalar], right: &[$scalar]) -> Vec<$bits> {
            left.iter()
                .zip(right)
                .map(|(lhs, rhs)| (*lhs $operator *rhs).to_bits())
                .collect()
        }

        fn verify_output(observed: &[$scalar], left: &[$scalar], right: &[$scalar]) {
            assert_eq!(observed.len(), left.len());
            assert_eq!(observed.len(), right.len());
            for ((actual, lhs), rhs) in observed.iter().zip(left).zip(right) {
                assert_eq!(actual.to_bits(), (*lhs $operator *rhs).to_bits());
            }
        }

        fn verify_bits(observed: &[$bits], left: &[$scalar], right: &[$scalar]) {
            assert_eq!(observed.len(), left.len());
            assert_eq!(observed.len(), right.len());
            for ((actual, lhs), rhs) in observed.iter().zip(left).zip(right) {
                assert_eq!(*actual, (*lhs $operator *rhs).to_bits());
            }
        }

        fn verify_source_fault(
            result: Result<PcuTensor<$scalar>, PcuExecutionError>,
            kind: PcuExecutionFaultKind,
        ) {
            match result {
                Err(PcuExecutionError::ArithmeticFault(fault)) => {
                    assert_eq!(fault.kind, kind);
                    assert_eq!(fault.invocation_id, 1);
                }
                other => panic!("expected source checked-float fault {kind:?}, got {other:?}"),
            }
        }

        #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
        #[allow(unsafe_code)]
        fn preflight_faults(
            backend: &RocmOwnedDispatchBackend,
            runtime: &HipRuntime,
            prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
            native_kernel: &HipKernel,
            native_stream: &fusion_pcu_rocm::HipStreamHandle,
            pcu_left: &mut DeviceBuffer,
            pcu_right: &mut DeviceBuffer,
            native_left: &mut DeviceBuffer,
            native_right: &mut DeviceBuffer,
            elements: usize,
            grid: u32,
        ) -> Result<(), Box<dyn Error>> {
            for (kind, left_bits, right_bits, expected_tag) in [
                (
                    PcuExecutionFaultKind::InvalidFloatingOperand,
                    $scalar::NAN.to_bits(),
                    $scalar::to_bits(1.0),
                    5u64,
                ),
                (
                    PcuExecutionFaultKind::ArithmeticOverflow,
                    $scalar::MAX.to_bits(),
                    overflow_rhs().to_bits(),
                    3u64,
                ),
            ].into_iter().chain(
                matches!(PcuDispatchFloatBinaryOp::$operation, PcuDispatchFloatBinaryOp::Div)
                    .then_some((PcuExecutionFaultKind::DivideByZero, $scalar::to_bits(0.0), $scalar::to_bits(0.0), 1u64))
            ) {
                let mut left = vec![1.0; elements];
                let mut right = vec![1.0; elements];
                left[0] = $scalar::from_bits(left_bits);
                right[0] = $scalar::from_bits(right_bits);
                pcu_left.copy_from(bytemuck::cast_slice(&left))?;
                pcu_right.copy_from(bytemuck::cast_slice(&right))?;
                native_left.copy_from(bytemuck::cast_slice(&left))?;
                native_right.copy_from(bytemuck::cast_slice(&right))?;
                let output = backend.allocate(elements * size_of::<$scalar>())?;
                let bindings = [
                    backend.binding(
                        PcuBindingRef::new(0, 0),
                        PcuBindingAccess::ReadOnly,
                        PcuBindingType::Value(PcuValueType::$scalar()),
                        (*pcu_left).clone(),
                    )?,
                    backend.binding(
                        PcuBindingRef::new(0, 1),
                        PcuBindingAccess::ReadOnly,
                        PcuBindingType::Value(PcuValueType::$scalar()),
                        (*pcu_right).clone(),
                    )?,
                    backend.binding(
                        PcuBindingRef::new(0, 2),
                        PcuBindingAccess::WriteOnly,
                        PcuBindingType::Value(PcuValueType::$scalar()),
                        output,
                    )?,
                ];
                let mut completion = prepared.submit(&bindings)?;
                assert_eq!(
                    completion.wait()?,
                    PcuCompletionOutcome::Fault(PcuExecutionFault {
                            recovered: false,
                        kind,
                        invocation_id: 0
                    })
                );
                drop(completion);

                let native_output = runtime.allocate(elements * size_of::<$scalar>())?;
                let mut native_fault = runtime.allocate(8)?;
                native_fault.copy_from(&u64::MAX.to_le_bytes())?;
                let args = [
                    HipKernelArgument::Buffer(native_left),
                    HipKernelArgument::Buffer(native_right),
                    HipKernelArgument::Buffer(&native_output),
                    HipKernelArgument::Buffer(&native_fault),
                ];
                // SAFETY: Each buffer covers `elements` floating-point values and the fault word is 8 bytes.
                let mut native_completion =
                    unsafe { native_kernel.launch(native_stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
                native_completion.wait()?;
                drop(native_completion);
                let mut bytes = [0_u8; 8];
                native_fault.copy_to(&mut bytes)?;
                assert_eq!(u64::from_le_bytes(bytes), expected_tag);
            }
            let mut left = vec![1.0; elements];
            let mut right = vec![1.0; elements];
            left[0] = $scalar::from_bits(1);
            right[0] = $scalar::from_bits(1);
            pcu_left.copy_from(bytemuck::cast_slice(&left))?;
            pcu_right.copy_from(bytemuck::cast_slice(&right))?;
            native_left.copy_from(bytemuck::cast_slice(&left))?;
            native_right.copy_from(bytemuck::cast_slice(&right))?;
            let output = backend.allocate(elements * size_of::<$scalar>())?;
            let bindings = [
                backend.binding(
                    PcuBindingRef::new(0, 0),
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::$scalar()),
                    (*pcu_left).clone(),
                )?,
                backend.binding(
                    PcuBindingRef::new(0, 1),
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::$scalar()),
                    (*pcu_right).clone(),
                )?,
                backend.binding(
                    PcuBindingRef::new(0, 2),
                    PcuBindingAccess::WriteOnly,
                    PcuBindingType::Value(PcuValueType::$scalar()),
                    output.clone(),
                )?,
            ];
            let mut completion = prepared.submit(&bindings)?;
            assert_eq!(completion.wait()?, PcuCompletionOutcome::Succeeded);
            drop(completion);
            let mut output_bytes = vec![0_u8; elements * size_of::<$scalar>()];
            output.copy_to(&mut output_bytes)?;
            assert_eq!($bits::from_ne_bytes(output_bytes[..size_of::<$scalar>()].try_into()?), ($scalar::from_bits(1) $operator $scalar::from_bits(1)).to_bits());

            let native_output = runtime.allocate(elements * size_of::<$scalar>())?;
            let mut native_fault = runtime.allocate(8)?;
            native_fault.copy_from(&u64::MAX.to_le_bytes())?;
            let args = [
                HipKernelArgument::Buffer(native_left),
                HipKernelArgument::Buffer(native_right),
                HipKernelArgument::Buffer(&native_output),
                HipKernelArgument::Buffer(&native_fault),
            ];
            // SAFETY: Each buffer covers `elements` floating-point values and the fault word is 8 bytes.
            let mut native_completion =
                unsafe { native_kernel.launch(native_stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
            native_completion.wait()?;
            drop(native_completion);
            let mut status = [0_u8; 8];
            native_fault.copy_to(&mut status)?;
            assert_eq!(u64::from_le_bytes(status), u64::MAX);
            let mut native_output_bytes = [0_u8; size_of::<$scalar>()];
            native_output.copy_to(&mut native_output_bytes)?;
            assert_eq!($bits::from_ne_bytes(native_output_bytes), ($scalar::from_bits(1) $operator $scalar::from_bits(1)).to_bits());
            Ok(())
        }

    };
}

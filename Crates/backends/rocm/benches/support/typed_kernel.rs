//! Shared host-slice and resident-buffer benchmark implementation.

#[path = "alloc.rs"]
#[allow(dead_code)] // The shared allocator helper also exposes phased counters for other benches.
mod alloc;

#[rustfmt::skip]
use std::{
    any::Any,
    error::Error,
    hint::black_box,
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
    PcuBindingRef,
    PcuDeviceBuffer,
    PcuExecutionError,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuMemoryPoolId,
};
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    RocmDiscovery,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmHostKernelError,
    compile_hip_source,
    lower_dispatch_to_hip_source,
};

#[pcu(invocations = N)]
fn transform<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 2.0 + 1.0;
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
    let (backend, selected) = crate::support::selection::open_ranked(&discovery, candidates, 256)?;
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        device: Some(selected.device.id),
        ..fusion_pcu::global::PcuExecutionPolicy::default()
    })?;
    let architecture = selected
        .architecture
        .as_deref()
        .ok_or("selected device has no HIP architecture")?;
    let runtime = discovery.open_device(selected.device)?;
    println!(
        "Typed kernel benchmark device: {} ({architecture})",
        selected.name
    );

    run_case::<65>(
        criterion,
        &backend,
        &runtime,
        &selected.name,
        selected.pool,
        architecture,
    )?;
    run_case::<1_048_576>(
        criterion,
        &backend,
        &runtime,
        &selected.name,
        selected.pool,
        architecture,
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_case<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    device: &str,
    pool: PcuMemoryPoolId,
    architecture: &str,
) -> Result<(), Box<dyn Error>> {
    println!("Typed kernel shape {N} on {device}");
    println!(
        "Matched checked complete-writer host boundary: input upload, fault reset, launch, terminal wait, fault status and output readback; neither route uploads old output contents."
    );
    let mut host_call = crate::support::cold_once("typed PCU host preparation", || {
        transform_prepare::<N, _>(backend)
    })?;
    let mut resident_call = crate::support::cold_once("typed PCU resident preparation", || {
        transform_prepare_device::<N, _>(backend)
    })?;
    let bindings = transform_bindings();
    let native_ir = transform_ir::<N>(&bindings)?;
    let native_source = native_ir
        .with_ir(lower_dispatch_to_hip_source)?
        .replace("fusion_kernel", "native_transform");
    let image = crate::support::cold_once("native HIP compilation", || {
        compile_hip_source(&native_source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let function = module.function(c"native_transform")?;
    let stream = runtime.create_stream()?;
    let mut host_input = Vec::with_capacity(N);
    let mut cpu_output = vec![0.0_f32; N];
    let initial_output = vec![-5.0_f32; N];
    let mut host_output = initial_output.clone();
    let native_input = runtime.allocate(N * core::mem::size_of::<f32>())?;
    let native_output = runtime.allocate(N * core::mem::size_of::<f32>())?;
    let native_fault = runtime.allocate(core::mem::size_of::<u64>())?;
    let mut native_result = vec![0.0_f32; N];
    let grid = u32::try_from(N)?.div_ceil(256);
    let mut native = NativeContext {
        input: native_input,
        output: native_output,
        fault: native_fault,
        fault_word_is_sentinel: false,
        function,
        stream,
        grid,
    };

    preflight_checked_faults::<N, _>(
        &mut host_call,
        &mut native,
        &mut host_output,
        &mut native_result,
    )?;

    preflight_host::<N, _>(&mut host_call, &mut host_output, &mut cpu_output)?;
    sample_host::<N, _>(
        criterion,
        &mut host_call,
        &mut host_input,
        &mut host_output,
        &mut cpu_output,
        &initial_output,
        &mut native,
    )?;
    balanced_host_diagnostic::<N, _>(
        &mut host_call,
        &mut host_input,
        &mut host_output,
        &mut cpu_output,
        &initial_output,
        &mut native,
    )?;
    allocation_diagnostic::<N, _>(
        &mut host_call,
        &mut host_output,
        &initial_output,
        &mut native,
        &mut native_result,
    )?;

    println!("Direct facade allocation control:");
    allocation_diagnostic::<N, _>(
        &mut |input: &[f32], output: &mut [f32]| transform::<N>(input, output),
        &mut host_output,
        &initial_output,
        &mut native,
        &mut native_result,
    )?;
    let mut resident_input = backend.upload_buffer(pool, &fresh_input::<N>(0))?;
    let mut resident_output = backend.upload_buffer(pool, &initial_output)?;
    // Warm up the reusable paths and inspect benchmark-thread Rust allocations around a call.
    host_call(&fresh_input::<N>(1), &mut host_output)?;
    backend.refresh_buffer(pool, &mut resident_input, &fresh_input::<N>(1))?;
    let capture = alloc::AllocationCapture::start();
    resident_call(&resident_input, &mut resident_output)?;
    let resident_allocations = alloc::AllocationCapture::finish();
    println!(
        "Prepared resident PCU warm call Rust heap: {} alloc/realloc calls, {} requested B; this is a benchmark-thread Rust allocator count, not a device/driver allocation count",
        resident_allocations.alloc_calls + resident_allocations.realloc_calls,
        resident_allocations.requested_bytes
    );
    drop(capture);

    sample_resident::<N, _>(
        criterion,
        &mut resident_call,
        backend,
        pool,
        &mut resident_input,
        &mut resident_output,
        &mut native,
        &mut native_result,
        &mut cpu_output,
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn preflight_host<const N: usize, E: Error + 'static>(
    call: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    output: &mut [f32],
    oracle: &mut [f32],
) -> Result<(), Box<dyn Error>> {
    let input = fresh_input::<N>(0);
    fill_oracle(&input, oracle);
    call(&input, output).map_err(|error| Box::new(error) as Box<dyn Error>)?;
    verify("host preflight", output, oracle)?;
    Ok(())
}

fn preflight_checked_faults<const N: usize, E: Error + 'static>(
    prepared: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    native: &mut NativeContext,
    source_output: &mut [f32],
    native_output: &mut [f32],
) -> Result<(), Box<dyn Error>> {
    for (input_value, kind) in [
        (f32::NAN, PcuExecutionFaultKind::InvalidFloatingOperand),
        (f32::MAX, PcuExecutionFaultKind::ArithmeticOverflow),
    ] {
        let mut input = vec![1.0_f32; N];
        input[1] = input_value;

        let source_error = transform::<N>(&input, source_output)
            .expect_err("source checked transform must surface arithmetic faults");
        let PcuExecutionError::ArithmeticFault(source_fault) = source_error else {
            panic!("expected source checked-float fault, got {source_error}");
        };
        assert_fault(source_fault, kind, 1);

        let prepared_error = prepared(&input, source_output)
            .expect_err("prepared checked transform must surface arithmetic faults");
        assert_prepared_fault(&prepared_error, kind, 1);

        let native_error = native_host_call(&input, native_output, native)
            .expect_err("native checked transform must surface arithmetic faults");
        let native_error = native_error
            .downcast::<PcuExecutionError>()
            .unwrap_or_else(|error| panic!("expected native checked-float fault, got {error}"));
        let PcuExecutionError::ArithmeticFault(native_fault) = *native_error else {
            panic!("expected native checked-float fault");
        };
        assert_fault(native_fault, kind, 1);
    }

    let input = fresh_input::<N>(0);
    let mut expected = vec![0.0_f32; N];
    fill_oracle(&input, &mut expected);
    transform::<N>(&input, source_output).expect("source checked retry succeeds");
    prepared(&input, source_output).map_err(|error| Box::new(error) as Box<dyn Error>)?;
    native_host_call(&input, native_output, native)?;
    verify("source checked retry", source_output, &expected)?;
    verify("native checked retry", native_output, &expected)?;
    Ok(())
}

fn assert_fault(fault: PcuExecutionFault, kind: PcuExecutionFaultKind, invocation_id: u64) {
    assert_eq!(fault.kind, kind);
    assert_eq!(fault.invocation_id, invocation_id);
}

fn assert_prepared_fault<E: Error + 'static>(
    error: &E,
    kind: PcuExecutionFaultKind,
    invocation_id: u64,
) {
    let Some(RocmHostKernelError::CheckedExecutionFault(fault)) =
        (error as &dyn Any).downcast_ref::<RocmHostKernelError>()
    else {
        panic!("expected prepared checked-float fault, got {error}");
    };
    assert_fault(*fault, kind, invocation_id);
}

#[allow(clippy::too_many_lines)]
#[allow(unsafe_code)]
// These borrow-only fixtures are clearer separately than nested in one mutable context object.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::significant_drop_tightening)] // `finish` is required to emit Criterion's group.
fn sample_host<const N: usize, E: Error + 'static>(
    criterion: &mut Criterion,
    call: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    input: &mut Vec<f32>,
    output: &mut [f32],
    oracle: &mut [f32],
    initial: &[f32],
    native: &mut NativeContext,
) -> Result<(), Box<dyn Error>> {
    // Cold facade resolution/compilation must not enter the warm Criterion interval.
    transform::<N>(&fresh_input::<N>(0), output)?;
    let mut group = criterion.benchmark_group(format!("typed-host-{N}"));
    group.throughput(Throughput::Elements(u64::try_from(N)?));
    for target in 0..3 {
        let label = match target {
            0 => "Prepared typed PCU",
            1 => "Direct typed PCU",
            _ => "Native HIP full boundary",
        };
        let mut sequence = 0_u64;
        group.bench_function(BenchmarkId::new(label, N), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let job = sequence;
                    sequence = sequence.saturating_add(1);
                    *input = fresh_input::<N>(job);
                    fill_oracle(input, oracle);
                    // Both routes begin from the same initialized mutable host output.
                    output.copy_from_slice(initial);
                    let started = Instant::now();
                    if target == 0 {
                        call(input, &mut *output).expect("typed host call");
                    } else if target == 1 {
                        transform::<N>(input, &mut *output).expect("direct typed host call");
                    } else {
                        native_host_call(input, &mut *output, native)
                            .expect("native HIP full host boundary");
                    }
                    elapsed += started.elapsed();
                    assert!(
                        verify(label, output, oracle).is_ok(),
                        "{label} output differs from CPU oracle at job {job}"
                    );
                    black_box(&*output);
                }
                elapsed
            });
        });
    }
    group.finish();
    Ok(())
}

struct NativeContext {
    input: DeviceBuffer,
    output: DeviceBuffer,
    fault: DeviceBuffer,
    fault_word_is_sentinel: bool,
    function: HipKernel,
    stream: HipStreamHandle,
    grid: u32,
}

impl NativeContext {
    #[allow(unsafe_code)]
    fn launch_and_check_fault(&mut self) -> Result<(), Box<dyn Error>> {
        let reset = !self.fault_word_is_sentinel;
        // Consume the proof before any launch attempt; only terminal status MAX restores it.
        self.fault_word_is_sentinel = false;
        if reset {
            self.fault.copy_from(&u64::MAX.to_le_bytes())?;
        }
        let arguments = [
            HipKernelArgument::Buffer(&self.input),
            HipKernelArgument::Buffer(&self.output),
            HipKernelArgument::Buffer(&self.fault),
        ];
        // SAFETY: Generated transform IR declares two f32 buffers plus an 8-byte checked fault
        // word; direct invocation indices are bounded by the validated logical extent.
        let mut completion = unsafe {
            self.function
                .launch(&self.stream, [self.grid, 1, 1], [256, 1, 1], 0, &arguments)
        }?;
        completion.wait()?;
        drop(completion);

        let mut fault_bytes = [0_u8; core::mem::size_of::<u64>()];
        self.fault.copy_to(&mut fault_bytes)?;
        let word = u64::from_le_bytes(fault_bytes);
        decode_native_fault(word)?;
        self.fault_word_is_sentinel = true;
        Ok(())
    }

    fn readback(&self, output: &mut [f32]) -> Result<(), Box<dyn Error>> {
        let mut result = PcuHostArgument::read_write(PcuBindingRef::new(0, 0), output);
        self.output
            .copy_to(result.bytes_mut().expect("mutable output bytes"))?;
        Ok(())
    }

    fn launch_and_readback(&mut self, output: &mut [f32]) -> Result<(), Box<dyn Error>> {
        self.launch_and_check_fault()?;
        self.readback(output)
    }
}

fn native_host_call(
    input: &[f32],
    output: &mut [f32],
    native: &mut NativeContext,
) -> Result<(), Box<dyn Error>> {
    let input_view = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
    native.input.copy_from(input_view.bytes())?;
    native.launch_and_readback(output)
}

fn decode_native_fault(word: u64) -> Result<(), Box<dyn Error>> {
    if word == u64::MAX {
        return Ok(());
    }
    let kind = match word & 0b111 {
        1 => PcuExecutionFaultKind::DivideByZero,
        2 => PcuExecutionFaultKind::SignedDivisionOverflow,
        3 => PcuExecutionFaultKind::ArithmeticOverflow,
        4 => PcuExecutionFaultKind::ArithmeticUnderflow,
        5 => PcuExecutionFaultKind::InvalidFloatingOperand,
        _ => return Err(format!("invalid native checked-fault word {word:#x}").into()),
    };
    Err(Box::new(PcuExecutionError::ArithmeticFault(
        PcuExecutionFault {
            recovered: false,
            kind,
            invocation_id: word >> 3,
        },
    )))
}

// The borrowed fixtures are the inputs to the paired routes and stay explicit at this boundary.
#[allow(clippy::too_many_arguments)]
fn balanced_host_diagnostic<const N: usize, E: Error + 'static>(
    prepared: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    input: &mut Vec<f32>,
    output: &mut [f32],
    oracle: &mut [f32],
    initial: &[f32],
    native: &mut NativeContext,
) -> Result<(), Box<dyn Error>> {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    const PAIRS: usize = 36;
    let mut prepared_times = [Duration::ZERO; PAIRS];
    let mut direct_times = [Duration::ZERO; PAIRS];
    let mut native_times = [Duration::ZERO; PAIRS];
    let mut prepared_native_ratios = [0.0_f64; PAIRS];
    let mut direct_native_ratios = [0.0_f64; PAIRS];

    for repeat in 0..6 {
        for (permutation_index, permutation) in PERMUTATIONS.into_iter().enumerate() {
            let sample_index = repeat * PERMUTATIONS.len() + permutation_index;
            let seed = 1_000_000 + u64::try_from(sample_index)?;
            *input = fresh_input::<N>(seed);
            fill_oracle(input, oracle);

            for route in permutation {
                // Host setup and restoration are outside the measured device call for every route.
                output.copy_from_slice(initial);
                let started = Instant::now();
                match route {
                    0 => prepared(input, output)
                        .map_err(|error| Box::new(error) as Box<dyn Error>)?,
                    1 => transform::<N>(input, output)?,
                    _ => native_host_call(input, output, native)?,
                }
                let elapsed = started.elapsed();
                match route {
                    0 => prepared_times[sample_index] = elapsed,
                    1 => direct_times[sample_index] = elapsed,
                    _ => native_times[sample_index] = elapsed,
                }
                verify("balanced host diagnostic", output, oracle)?;
                black_box(&*output);
            }
            prepared_native_ratios[sample_index] = prepared_times[sample_index].as_secs_f64()
                / native_times[sample_index].as_secs_f64();
            direct_native_ratios[sample_index] =
                direct_times[sample_index].as_secs_f64() / native_times[sample_index].as_secs_f64();
        }
    }

    println!(
        "Balanced host diagnostic ({N}, six permutations × six, 36 triples): prepared median {:?}, direct median {:?}, native median {:?}; prepared/native median {:.4}×, direct/native median {:.4}×. These are paired host-wall diagnostics, not Criterion confidence intervals.",
        median_duration(prepared_times),
        median_duration(direct_times),
        median_duration(native_times),
        median_f64(prepared_native_ratios),
        median_f64(direct_native_ratios),
    );
    Ok(())
}

fn median_duration(mut values: [Duration; 36]) -> Duration {
    values.sort_unstable();
    values[17]
        + values[18]
            .checked_sub(values[17])
            .expect("durations are sorted")
            / 2
}

fn median_f64(mut values: [f64; 36]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[17].midpoint(values[18])
}

#[allow(clippy::too_many_lines)]
#[allow(unsafe_code)]
// The sample owns one reusable mutable buffer set; splitting it obscures the paired work boundary.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::significant_drop_tightening)] // `finish` is required to emit Criterion's group.
fn sample_resident<const N: usize, E: Error + 'static>(
    criterion: &mut Criterion,
    call: &mut impl FnMut(
        &PcuDeviceBuffer<f32, RocmMemoryResource>,
        &mut PcuDeviceBuffer<f32, RocmMemoryResource>,
    ) -> Result<(), E>,
    backend: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    input: &mut PcuDeviceBuffer<f32, RocmMemoryResource>,
    output: &mut PcuDeviceBuffer<f32, RocmMemoryResource>,
    native: &mut NativeContext,
    native_result: &mut [f32],
    oracle: &mut [f32],
) -> Result<(), Box<dyn Error>> {
    let mut group = criterion.benchmark_group(format!("typed-resident-{N}"));
    group.throughput(Throughput::Elements(u64::try_from(N)?));
    for pcu_target in [true, false] {
        let label = if pcu_target {
            "Prepared typed PCU"
        } else {
            "Native HIP"
        };
        let mut sequence = 0_u64;
        group.bench_function(BenchmarkId::new(label, N), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let job = sequence;
                    sequence = sequence.saturating_add(1);
                    let input_values = fresh_input::<N>(job);
                    fill_oracle(&input_values, oracle);
                    if pcu_target {
                        backend
                            .refresh_buffer(pool, input, &input_values)
                            .expect("resident PCU input refresh");
                        let started = Instant::now();
                        call(input, output).expect("resident PCU call");
                        elapsed += started.elapsed();
                        backend
                            .download_buffer(pool, output, native_result)
                            .expect("resident PCU result download");
                    } else {
                        let input_view =
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input_values);
                        native
                            .input
                            .copy_from(input_view.bytes())
                            .expect("resident HIP input upload");
                        let started = Instant::now();
                        native
                            .launch_and_check_fault()
                            .expect("resident HIP checked launch and fault status");
                        elapsed += started.elapsed();
                        native
                            .readback(native_result)
                            .expect("resident HIP result download");
                    }
                    verify(label, native_result, oracle)
                        .unwrap_or_else(|_| panic!("{label} differs from CPU oracle at job {job}"));
                    black_box(&*native_result);
                }
                elapsed
            });
        });
    }
    group.finish();
    // Alternate adjacent routes to check whether sequential Criterion groups hide clock/order
    // effects. Resident payload readback and verification remain outside each timer.
    let mut pcu_walls = [Duration::ZERO; 32];
    let mut native_walls = [Duration::ZERO; 32];
    let mut paired_ratios = [0.0_f64; 32];
    for sample in 0..32 {
        let values = fresh_input::<N>(u64::try_from(sample)? + 1_000_000);
        fill_oracle(&values, oracle);
        for pcu_target in [sample % 2 == 0, sample % 2 != 0] {
            if pcu_target {
                backend.refresh_buffer(pool, input, &values)?;
                let started = Instant::now();
                call(input, output)?;
                pcu_walls[sample] = started.elapsed();
                backend.download_buffer(pool, output, native_result)?;
            } else {
                let view = PcuHostArgument::read(PcuBindingRef::new(0, 0), &values);
                native.input.copy_from(view.bytes())?;
                let started = Instant::now();
                native.launch_and_check_fault()?;
                native_walls[sample] = started.elapsed();
                native.readback(native_result)?;
            }
            verify("paired resident diagnostic", native_result, oracle)?;
            black_box(&*native_result);
        }
        paired_ratios[sample] =
            pcu_walls[sample].as_secs_f64() / native_walls[sample].as_secs_f64();
    }
    pcu_walls.sort_unstable();
    native_walls.sort_unstable();
    paired_ratios.sort_unstable_by(f64::total_cmp);
    println!(
        "Paired resident {N}, 32 alternating CPU-verified pairs: PCU {:?}, native {:?}, paired-ratio median {:.4}; diagnostic medians, not Criterion intervals",
        pcu_walls[15]
            + pcu_walls[16]
                .checked_sub(pcu_walls[15])
                .expect("durations sorted")
                / 2,
        native_walls[15]
            + native_walls[16]
                .checked_sub(native_walls[15])
                .expect("durations sorted")
                / 2,
        paired_ratios[15].midpoint(paired_ratios[16])
    );
    Ok(())
}

fn fresh_input<const N: usize>(iteration: u64) -> Vec<f32> {
    let phase_bits = 0x3f00_0000 | (u32::try_from(iteration).unwrap_or(u32::MAX) & 0x007f_ffff);
    let phase = f32::from_bits(phase_bits);
    (0..N)
        .map(|index| {
            let base = f32::from(u16::try_from(index % 1024).expect("sample fits")) * 0.25;
            base + phase
        })
        .collect()
}

fn fill_oracle(input: &[f32], output: &mut [f32]) {
    for (destination, source) in output.iter_mut().zip(input) {
        *destination = source * 2.0 + 1.0;
    }
}

fn verify(label: &str, actual: &[f32], expected: &[f32]) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(format!("{label} output differs from CPU oracle").into());
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
// This one-off diagnostic deliberately compares the same two host-call boundaries.
fn allocation_diagnostic<const N: usize, E: Error + 'static>(
    call: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    pcu_output: &mut [f32],
    initial: &[f32],
    native: &mut NativeContext,
    native_result: &mut [f32],
) -> Result<(), Box<dyn Error>> {
    let input = fresh_input::<N>(u64::MAX - 8);
    let mut expected = vec![0.0_f32; N];
    fill_oracle(&input, &mut expected);
    pcu_output.copy_from_slice(initial);
    let capture = alloc::AllocationCapture::start();
    call(&input, pcu_output).map_err(|error| Box::new(error) as Box<dyn Error>)?;
    let pcu_counts = alloc::AllocationCapture::finish();
    drop(capture);
    verify("PCU heap census", pcu_output, &expected)?;

    let capture = alloc::AllocationCapture::start();
    native_host_call(&input, native_result, native)?;
    let native_counts = alloc::AllocationCapture::finish();
    drop(capture);
    verify("native HIP heap census", native_result, &expected)?;
    println!(
        "Warm host-call Rust heap census (outside Criterion): PCU {} alloc/realloc calls, {} requested B; HIP {} calls, {} requested B. HIP/driver allocations are not visible to this Rust allocator counter.",
        pcu_counts.alloc_calls + pcu_counts.realloc_calls,
        pcu_counts.requested_bytes,
        native_counts.alloc_calls + native_counts.realloc_calls,
        native_counts.requested_bytes,
    );
    Ok(())
}

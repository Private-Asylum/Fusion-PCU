//! Explicit clamped execution compared with the same lowered HIP program.
//! Compilation/input generation/oracles stay outside the timer. Every timed route includes
//! input upload, terminal completion, status inspection and completed output download.
#[path = "alloc.rs"]
#[allow(dead_code)]
mod alloc;
#[rustfmt::skip]
use std::{
    any::Any,
    error::Error,
    hint::black_box,
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
    PcuBindingRef,
    PcuClampedError,
    PcuClampedFloat,
    PcuExecutionError,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuHostArgument,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    RocmDiscovery,
    RocmHostKernelError,
    RocmOwnedDispatchBackend,
    compile_hip_source,
    lower_dispatch_to_hip_source,
};
use fusion_pcu_macros::pcu;

#[pcu(invocations = N, flag(clamp_range))]
fn transform<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 2.0 / 2.0 + 1.0;
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = crate::support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect();
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
        "Explicit clamp benchmark: {} ({architecture}); native uses identical PCU-lowered HIP",
        selected.name
    );
    run_shape::<65>(criterion, &backend, &runtime, architecture)?;
    run_shape::<1_048_576>(criterion, &backend, &runtime, architecture)
}

struct Native {
    input: DeviceBuffer,
    output: DeviceBuffer,
    fault: DeviceBuffer,
    function: HipKernel,
    stream: HipStreamHandle,
    grid: u32,
    clean: bool,
}

impl Native {
    #[allow(unsafe_code)]
    fn call(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<Option<PcuExecutionFault>, Box<dyn Error>> {
        self.input
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), input).bytes())?;
        let reset = !self.clean;
        self.clean = false;
        if reset {
            self.fault.copy_from(&u64::MAX.to_le_bytes())?;
        }
        let arguments = [
            HipKernelArgument::Buffer(&self.input),
            HipKernelArgument::Buffer(&self.output),
            HipKernelArgument::Buffer(&self.fault),
        ];
        // SAFETY: The validated generated program has two F32 bindings, an 8-byte status
        // word, and a bounded N-element complete writer. All resources survive terminal wait.
        let mut completion = unsafe {
            self.function
                .launch(&self.stream, [self.grid, 1, 1], [256, 1, 1], 0, &arguments)
        }?;
        completion.wait()?;
        drop(completion);
        let mut bytes = [0_u8; 8];
        self.fault.copy_to(&mut bytes)?;
        let word = u64::from_le_bytes(bytes);
        let recovery = if word == u64::MAX {
            self.clean = true;
            None
        } else {
            if word & (1_u64 << 63) == 0 {
                return Err("native clamp benchmark encountered a fatal fault".into());
            }
            let kind = match word & 7 {
                3 => PcuExecutionFaultKind::ArithmeticOverflow,
                4 => PcuExecutionFaultKind::ArithmeticUnderflow,
                _ => return Err("invalid native recovered fault word".into()),
            };
            Some(PcuExecutionFault {
                kind,
                invocation_id: (word & !(1_u64 << 63)) >> 3,
                recovered: true,
            })
        };
        let mut view = PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output);
        self.output
            .copy_to(view.bytes_mut().expect("mutable result"))?;
        Ok(recovery)
    }
}

fn classify<E: Error + 'static>(result: Result<(), E>) -> Option<PcuExecutionFault> {
    match result {
        Ok(()) => None,
        Err(error) => {
            let erased = &error as &dyn Any;
            let fault = if let Some(PcuExecutionError::ArithmeticFault(fault)) =
                erased.downcast_ref::<PcuExecutionError>()
            {
                *fault
            } else if let Some(RocmHostKernelError::CheckedExecutionFault(fault)) =
                erased.downcast_ref::<RocmHostKernelError>()
            {
                *fault
            } else {
                panic!("unexpected clamp execution error: {error}");
            };
            assert!(fault.recovered, "clamp result must be complete");
            Some(fault)
        }
    }
}

fn timed<const N: usize, E: Error + 'static>(
    route: usize,
    prepared: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    native: &mut Native,
    input: &[f32],
    output: &mut [f32],
) -> (Duration, Option<PcuExecutionFault>) {
    let started = Instant::now();
    match route {
        0 => {
            let result = prepared(input, output);
            let duration = started.elapsed();
            (duration, classify(result))
        }
        1 => {
            let result = transform::<N>(input, output);
            let duration = started.elapsed();
            (duration, classify(result))
        }
        _ => {
            let result = native.call(input, output);
            let duration = started.elapsed();
            (duration, result.expect("native clamp execution"))
        }
    }
}

fn recover(result: Result<f32, PcuClampedError<f32>>) -> f32 {
    match result {
        Ok(value) => value,
        Err(PcuClampedError::Range(fault)) => fault.clamped_value(),
        Err(PcuClampedError::Fatal(kind)) => panic!("unexpected scalar oracle failure: {kind:?}"),
    }
}

fn fill(
    input: &mut [f32],
    oracle: &mut [f32],
    sequence: u64,
    faulting: bool,
) -> Option<PcuExecutionFault> {
    let offset = u32::try_from(sequence & 0x007f_ffff).expect("masked sequence");
    for (index, value) in input.iter_mut().enumerate() {
        let lane = u32::try_from(index & 0x007f_ffff).expect("masked lane");
        *value = f32::from_bits(0x3f00_0000 | ((lane + offset) & 0x007f_ffff));
    }
    let fault = if faulting {
        let index = usize::try_from(sequence % u64::try_from(input.len()).expect("host extent"))
            .expect("bounded lane");
        input[index] = if sequence & 1 == 0 {
            f32::MAX
        } else {
            -f32::MAX
        };
        Some(PcuExecutionFault {
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            invocation_id: u64::try_from(index).expect("host index"),
            recovered: true,
        })
    } else {
        None
    };
    for (value, expected) in input.iter().zip(oracle) {
        let scaled = recover(value.pcu_clamped_mul(2.0));
        let restored = recover(scaled.pcu_clamped_div(2.0));
        *expected = recover(restored.pcu_clamped_add(1.0));
    }
    fault
}

fn verify(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "complete output extent");
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "complete clamp output differs from scalar oracle"
    );
}

#[allow(clippy::too_many_lines)] // Keep setup, preflight and paired timing boundaries together.
fn run_shape<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    architecture: &str,
) -> Result<(), Box<dyn Error>> {
    let mut prepared = crate::support::cold_once("clamped host preparation", || {
        transform_prepare::<N, _>(backend)
    })?;
    let bindings = transform_bindings();
    let source = transform_ir::<N>(&bindings)?
        .with_ir(lower_dispatch_to_hip_source)?
        .replace("fusion_kernel", "native_clamped_transform");
    let image = crate::support::cold_once("clamped native compilation", || {
        compile_hip_source(&source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let mut native = Native {
        input: runtime.allocate(N * 4)?,
        output: runtime.allocate(N * 4)?,
        fault: runtime.allocate(8)?,
        function: module.function(c"native_clamped_transform")?,
        stream: runtime.create_stream()?,
        grid: u32::try_from(N)?.div_ceil(256),
        clean: false,
    };
    let mut input = vec![0.0; N];
    let mut output = vec![0.0; N];
    let mut oracle = vec![0.0; N];
    for faulting in [false, true] {
        let label = if faulting { "recovery" } else { "nominal" };
        let expected = fill(&mut input, &mut oracle, 0, faulting);
        assert_eq!(classify(prepared(&input, &mut output)), expected);
        verify(&output, &oracle);
        assert_eq!(classify(transform::<N>(&input, &mut output)), expected);
        verify(&output, &oracle);
        assert_eq!(native.call(&input, &mut output)?, expected);
        verify(&output, &oracle);
        {
            let mut group = criterion.benchmark_group(format!("clamped-f32-{label}-{N}"));
            group.throughput(Throughput::Elements(u64::try_from(N)?));
            for route in 0..3 {
                let name = ["Prepared PCU", "Direct PCU", "PCU-lowered HIP"][route];
                let expected = fill(&mut input, &mut oracle, 0, faulting);
                let capture = alloc::AllocationCapture::start();
                let actual = match route {
                    0 => classify(prepared(&input, &mut output)),
                    1 => classify(transform::<N>(&input, &mut output)),
                    _ => native.call(&input, &mut output)?,
                };
                let counts = alloc::AllocationCapture::finish();
                drop(capture);
                assert_eq!(actual, expected);
                verify(&output, &oracle);
                println!(
                    "{label}/{N}/{name}: {} Rust allocations, {} requested bytes",
                    counts.alloc_calls + counts.realloc_calls,
                    counts.requested_bytes
                );
                let mut sequence = 1_u64;
                group.bench_function(BenchmarkId::new(name, N), |bencher| {
                    bencher.iter_custom(|iterations| {
                        let mut elapsed = Duration::ZERO;
                        for _ in 0..iterations {
                            let expected = fill(&mut input, &mut oracle, sequence, faulting);
                            sequence = sequence.wrapping_add(1);
                            let (duration, actual) = timed::<N, _>(
                                route,
                                &mut prepared,
                                &mut native,
                                &input,
                                &mut output,
                            );
                            elapsed += duration;
                            assert_eq!(actual, expected);
                            verify(&output, &oracle);
                            black_box(&output);
                        }
                        elapsed
                    });
                });
            }
            group.finish();
        }
        let orders = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut times = [[Duration::ZERO; 36]; 3];
        let mut ratios = [[0.0_f64; 36]; 2];
        for repeat in 0..6 {
            for (position, order) in orders.into_iter().enumerate() {
                let sample = repeat * 6 + position;
                let expected = fill(
                    &mut input,
                    &mut oracle,
                    1_000_000 + u64::try_from(sample)?,
                    faulting,
                );
                for route in order {
                    let (duration, actual) =
                        timed::<N, _>(route, &mut prepared, &mut native, &input, &mut output);
                    times[route][sample] = duration;
                    assert_eq!(actual, expected);
                    verify(&output, &oracle);
                }
                for route in 0..2 {
                    ratios[route][sample] =
                        times[route][sample].as_secs_f64() / times[2][sample].as_secs_f64();
                }
            }
        }
        for row in &mut times {
            row.sort_unstable();
        }
        for row in &mut ratios {
            row.sort_by(f64::total_cmp);
        }
        println!(
            "Balanced {label}/{N}, 36 paired samples, median us prepared/direct/lowered: {:.3}/{:.3}/{:.3}; paired PCU/lowered ratios {:.4}/{:.4} (diagnostics, no confidence intervals)",
            (times[0][17].as_secs_f64() + times[0][18].as_secs_f64()) * 0.5e6,
            (times[1][17].as_secs_f64() + times[1][18].as_secs_f64()) * 0.5e6,
            (times[2][17].as_secs_f64() + times[2][18].as_secs_f64()) * 0.5e6,
            ratios[0][17].midpoint(ratios[0][18]),
            ratios[1][17].midpoint(ratios[1][18])
        );
    }
    Ok(())
}

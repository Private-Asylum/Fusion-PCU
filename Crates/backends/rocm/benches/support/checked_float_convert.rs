//! Prepared, direct, and native HIP routes for checked float conversions.

#[path = "alloc.rs"]
#[allow(dead_code)]
mod alloc;

#[rustfmt::skip]
use std::{
    error::Error,
    ffi::CStr,
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
    PcuCheckedFloatConversion,
    PcuCheckedFloatWidening,
    PcuExecutionError,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
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

macro_rules! checked_conversion_profile {
    ($module:ident, $kernel_module:ident, $kernel:ident, $prepare:ident, $ir:ident, $bindings:ident,
     $src:ty, $dst:ty, $fill:ident, $oracle:ident, $fault_cases:ident,
     $label:literal, $symbol:literal) => {
        mod $module {
            use super::*;
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
        "Checked {} conversion benchmark device: {} ({architecture})",
        $label, selected.name
    );
    run_case::<65>(criterion, &backend, &runtime, &selected.name, architecture)?;
    run_case::<1_048_576>(criterion, &backend, &runtime, &selected.name, architecture)?;
    Ok(())
}

fn run_case<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    device: &str,
    architecture: &str,
) -> Result<(), Box<dyn Error>> {
    println!("Checked {} shape {N} on {device}", $label);
    let mut prepared = crate::support::cold_once("checked conversion host preparation", || {
        super::$kernel_module::$prepare::<N, _>(backend)
    })?;
    let bindings = super::$kernel_module::$bindings();
    let source = super::$kernel_module::$ir::<N>(&bindings)?
        .with_ir(lower_dispatch_to_hip_source)?
        .replace("fusion_kernel", $symbol);
    let image = crate::support::cold_once("checked conversion native HIP compilation", || {
        compile_hip_source(&source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let function = module.function(CStr::from_bytes_with_nul(concat!($symbol, "\0").as_bytes())?)?;
    let stream = runtime.create_stream()?;
    let native_input = runtime.allocate(N * core::mem::size_of::<$src>())?;
    let native_output_buffer = runtime.allocate(N * core::mem::size_of::<$dst>())?;
    let fault = runtime.allocate(core::mem::size_of::<u64>())?;
    let initial = vec![<$dst>::default(); N];
    let mut output = initial.clone();
    let mut oracle = vec![<$dst>::default(); N];
    let mut input = vec![<$src>::default(); N];
    let mut native_output_values = vec![<$dst>::default(); N];
    let mut native = NativeContext {
        input: native_input,
        output: native_output_buffer,
        fault,
        function,
        stream,
        grid: u32::try_from(N)?.div_ceil(256),
        fault_is_sentinel: false,
    };

    preflight_faults::<N, _>(
        &mut prepared,
        &mut native,
        &mut output,
        &mut native_output_values,
    )?;
    super::$fill(&mut input, 0);
    super::$oracle(&input, &mut oracle)?;
    call_all_routes::<N, _>(
        &mut prepared,
        &mut native,
        &input,
        &mut output,
        &mut native_output_values,
        &initial,
        &oracle,
    )?;
    allocation_diagnostic::<N, _>(
        &mut prepared,
        &mut native,
        &input,
        &mut output,
        &mut native_output_values,
        &initial,
        &oracle,
    )?;
    sample_host::<N, _>(
        criterion,
        &mut prepared,
        &mut native,
        &mut input,
        &mut output,
        &mut native_output_values,
        &initial,
        &mut oracle,
    )?;
    balanced_diagnostic::<N, _>(
        &mut prepared,
        &mut native,
        &mut input,
        &mut output,
        &mut native_output_values,
        &initial,
        &mut oracle,
    )?;
    Ok(())
}

fn verify(label: &str, actual: &[$dst], expected: &[$dst]) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(got, wanted)| got.to_bits() != wanted.to_bits())
    {
        return Err(format!("{label} result differs from checked-conversion oracle").into());
    }
    Ok(())
}

struct NativeContext {
    input: DeviceBuffer,
    output: DeviceBuffer,
    fault: DeviceBuffer,
    function: HipKernel,
    stream: HipStreamHandle,
    grid: u32,
    fault_is_sentinel: bool,
}

impl NativeContext {
    #[allow(unsafe_code)]
    fn launch_and_check_fault(&mut self) -> Result<(), Box<dyn Error>> {
        let reset = !self.fault_is_sentinel;
        self.fault_is_sentinel = false;
        if reset {
            self.fault.copy_from(&u64::MAX.to_le_bytes())?;
        }
        let arguments = [
            HipKernelArgument::Buffer(&self.input),
            HipKernelArgument::Buffer(&self.output),
            HipKernelArgument::Buffer(&self.fault),
        ];
        // SAFETY: Generated bindings describe the selected profile source/destination and its
        // 8-byte fault word; invocation indices are bounded by the validated logical extent.
        let mut completion = unsafe {
            self.function
                .launch(&self.stream, [self.grid, 1, 1], [256, 1, 1], 0, &arguments)
        }?;
        completion.wait()?;
        drop(completion);
        let mut bytes = [0_u8; core::mem::size_of::<u64>()];
        self.fault.copy_to(&mut bytes)?;
        decode_fault(u64::from_le_bytes(bytes))?;
        self.fault_is_sentinel = true;
        Ok(())
    }

    fn readback(&self, output: &mut [$dst]) -> Result<(), Box<dyn Error>> {
        let mut view = PcuHostArgument::read_write(PcuBindingRef::new(0, 0), output);
        self.output
            .copy_to(view.bytes_mut().expect("mutable output bytes"))?;
        Ok(())
    }

    fn full_host_call(&mut self, input: &[$src], output: &mut [$dst]) -> Result<(), Box<dyn Error>> {
        let view = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
        self.input.copy_from(view.bytes())?;
        self.launch_and_check_fault()?;
        self.readback(output)
    }
}

fn decode_fault(word: u64) -> Result<(), Box<dyn Error>> {
    if word == u64::MAX {
        return Ok(());
    }
    let kind = match word & 0b111 {
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

fn check_fault(error: &(dyn Error + 'static), kind: PcuExecutionFaultKind, id: u64) {
    if let Some(PcuExecutionError::ArithmeticFault(fault)) =
        error.downcast_ref::<PcuExecutionError>()
    {
        assert_eq!(fault.kind, kind);
        assert_eq!(fault.invocation_id, id);
        return;
    }
    if let Some(RocmHostKernelError::CheckedExecutionFault(fault)) =
        error.downcast_ref::<RocmHostKernelError>()
    {
        assert_eq!(fault.kind, kind);
        assert_eq!(fault.invocation_id, id);
        return;
    }
    panic!("expected checked conversion fault {kind:?} at {id}, got {error}");
}

fn preflight_faults<const N: usize, E: Error + 'static>(
    prepared: &mut impl FnMut(&[$src], &mut [$dst]) -> Result<(), E>,
    native: &mut NativeContext,
    output: &mut [$dst],
    native_output: &mut [$dst],
) -> Result<(), Box<dyn Error>> {
    let retained_output = output.to_vec();
    let retained_native_output = native_output.to_vec();
    for (bad_value, expected) in super::$fault_cases() {
        let mut input = vec![<$src>::default(); N];
        super::$fill(&mut input, 7);
        input[1] = bad_value;
        output.copy_from_slice(&retained_output);
        check_fault(
            &super::$kernel_module::$kernel::<N>(&input, output)
                .expect_err("direct conversion faults"),
            expected,
            1,
        );
        assert_eq!(output, retained_output, "direct fault changed host output");
        output.copy_from_slice(&retained_output);
        check_fault(
            &prepared(&input, output).expect_err("prepared conversion faults"),
            expected,
            1,
        );
        assert_eq!(
            output, retained_output,
            "prepared fault changed host output"
        );
        native_output.copy_from_slice(&retained_native_output);
        let native_error = native
            .full_host_call(&input, native_output)
            .expect_err("native conversion faults");
        check_fault(native_error.as_ref(), expected, 1);
        assert_eq!(
            native_output, retained_native_output,
            "native fault read back partial device output"
        );
    }
    let mut retry = vec![<$src>::default(); N];
    super::$fill(&mut retry, 33);
    let mut expected = vec![<$dst>::default(); N];
    super::$oracle(&retry, &mut expected)?;
    super::$kernel_module::$kernel::<N>(&retry, output)?;
    verify("direct retry", output, &expected)?;
    prepared(&retry, output).map_err(|error| Box::new(error) as Box<dyn Error>)?;
    verify("prepared retry", output, &expected)?;
    native.full_host_call(&retry, native_output)?;
    verify("native retry", native_output, &expected)?;
    Ok(())
}

fn call_all_routes<const N: usize, E: Error + 'static>(
    prepared: &mut impl FnMut(&[$src], &mut [$dst]) -> Result<(), E>,
    native: &mut NativeContext,
    input: &[$src],
    output: &mut [$dst],
    native_output: &mut [$dst],
    initial: &[$dst],
    oracle: &[$dst],
) -> Result<(), Box<dyn Error>> {
    output.copy_from_slice(initial);
    super::$kernel_module::$kernel::<N>(input, output)?;
    verify("direct preflight", output, oracle)?;
    output.copy_from_slice(initial);
    prepared(input, output).map_err(|error| Box::new(error) as Box<dyn Error>)?;
    verify("prepared preflight", output, oracle)?;
    native.full_host_call(input, native_output)?;
    verify("native preflight", native_output, oracle)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::significant_drop_tightening)] // `finish` closes the paired Criterion route group.
fn sample_host<const N: usize, E: Error + 'static>(
    criterion: &mut Criterion,
    prepared: &mut impl FnMut(&[$src], &mut [$dst]) -> Result<(), E>,
    native: &mut NativeContext,
    input: &mut [$src],
    output: &mut [$dst],
    native_output: &mut [$dst],
    initial: &[$dst],
    oracle: &mut [$dst],
) -> Result<(), Box<dyn Error>> {
    let mut group = criterion.benchmark_group(format!("checked-{}-host-{N}", $label));
    group.throughput(Throughput::Elements(u64::try_from(N)?));
    for target in 0..3 {
        let name = [
            "Prepared checked PCU",
            "Direct checked PCU",
            "Native HIP full boundary",
        ][target];
        let mut sequence = 0_u64;
        group.bench_function(BenchmarkId::new(name, N), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    super::$fill(input, sequence);
                    sequence = sequence.saturating_add(1);
                    super::$oracle(input, oracle).expect("checked conversion oracle");
                    output.copy_from_slice(initial);
                    let started = Instant::now();
                    match target {
                        0 => prepared(input, output).expect("prepared conversion"),
                        1 => super::$kernel_module::$kernel::<N>(input, output)
                            .expect("direct conversion"),
                        _ => native
                            .full_host_call(input, native_output)
                            .expect("native full conversion boundary"),
                    }
                    elapsed += started.elapsed();
                    let actual = if target == 2 {
                        &*native_output
                    } else {
                        &*output
                    };
                    assert!(verify(name, actual, oracle).is_ok(), "{name} mismatch");
                    black_box(actual);
                }
                elapsed
            });
        });
    }
    group.finish();
    Ok(())
}

fn balanced_diagnostic<const N: usize, E: Error + 'static>(
    prepared: &mut impl FnMut(&[$src], &mut [$dst]) -> Result<(), E>,
    native: &mut NativeContext,
    input: &mut [$src],
    output: &mut [$dst],
    native_output: &mut [$dst],
    initial: &[$dst],
    oracle: &mut [$dst],
) -> Result<(), Box<dyn Error>> {
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut times = [[Duration::ZERO; 36]; 3];
    for repeat in 0..6 {
        for (perm_index, permutation) in PERMS.into_iter().enumerate() {
            let sample = repeat * 6 + perm_index;
            super::$fill(input, 1_000_000 + u64::try_from(sample)?);
            super::$oracle(input, oracle)?;
            for route in permutation {
                output.copy_from_slice(initial);
                let started = Instant::now();
                match route {
                    0 => prepared(input, output)
                        .map_err(|error| Box::new(error) as Box<dyn Error>)?,
                    1 => super::$kernel_module::$kernel::<N>(input, output)?,
                    _ => native.full_host_call(input, native_output)?,
                }
                times[route][sample] = started.elapsed();
                let actual = if route == 2 {
                    &*native_output
                } else {
                    &*output
                };
                verify("balanced conversion diagnostic", actual, oracle)?;
                black_box(actual);
            }
        }
    }
    let medians = times.map(median_duration);
    let prepared_native_ratios = core::array::from_fn(|sample| {
        times[0][sample].as_secs_f64() / times[2][sample].as_secs_f64()
    });
    let direct_native_ratios = core::array::from_fn(|sample| {
        times[1][sample].as_secs_f64() / times[2][sample].as_secs_f64()
    });
    println!(
        "Balanced {} host diagnostic ({N}, six permutations × six, 36 triples): prepared {:?}, direct {:?}, native {:?}; median per-triple prepared/native {:.3}×, direct/native {:.3}×; diagnostic only, not Criterion confidence intervals",
        $label, medians[0], medians[1], medians[2],
        median_f64(prepared_native_ratios), median_f64(direct_native_ratios),
    );
    Ok(())
}

fn median_f64(mut values: [f64; 36]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    (values[17] + values[18]) / 2.0
}

fn median_duration(mut values: [Duration; 36]) -> Duration {
    values.sort_unstable();
    values[17]
        + values[18]
            .checked_sub(values[17])
            .expect("durations sorted")
            / 2
}

fn allocation_diagnostic<const N: usize, E: Error + 'static>(
    prepared: &mut impl FnMut(&[$src], &mut [$dst]) -> Result<(), E>,
    native: &mut NativeContext,
    input: &[$src],
    output: &mut [$dst],
    native_output: &mut [$dst],
    initial: &[$dst],
    oracle: &[$dst],
) -> Result<(), Box<dyn Error>> {
    output.copy_from_slice(initial);
    let capture = alloc::AllocationCapture::start();
    prepared(input, output).map_err(|error| Box::new(error) as Box<dyn Error>)?;
    let pcu = alloc::AllocationCapture::finish();
    drop(capture);
    verify("prepared conversion heap census", output, oracle)?;

    output.copy_from_slice(initial);
    let capture = alloc::AllocationCapture::start();
    super::$kernel_module::$kernel::<N>(input, output)?;
    let direct = alloc::AllocationCapture::finish();
    drop(capture);
    verify("direct conversion heap census", output, oracle)?;

    let capture = alloc::AllocationCapture::start();
    native.full_host_call(input, native_output)?;
    let hip = alloc::AllocationCapture::finish();
    drop(capture);
    verify("native conversion heap census", native_output, oracle)?;
    println!(
        "Warm checked {} Rust heap census (outside Criterion): prepared {} alloc/realloc, {} requested B; direct {} calls, {} B; native {} calls, {} B. HIP/driver allocations are invisible to this Rust allocator counter.",
        $label,
        pcu.alloc_calls + pcu.realloc_calls,
        pcu.requested_bytes,
        direct.alloc_calls + direct.realloc_calls,
        direct.requested_bytes,
        hip.alloc_calls + hip.realloc_calls,
        hip.requested_bytes,
    );
    Ok(())
}

        }
    };
}

fn narrow_fill(input: &mut [f64], seed: u64) {
    for (index, value) in input.iter_mut().enumerate() {
        let step = u64::try_from(index).unwrap_or(u64::MAX).wrapping_mul(17);
        let mantissa =
            u32::try_from(seed.wrapping_add(step) & 0x000f_ffff).expect("masked significand fits");
        let fractional = f64::from_bits((1023_u64 << 52) | (u64::from(mantissa) << 20)) - 1.0;
        *value = f64::from(u16::try_from(index % 2048).expect("sample fits")) * 0.125 + fractional;
    }
}
fn narrow_oracle(input: &[f64], output: &mut [f32]) -> Result<(), Box<dyn Error>> {
    for (destination, source) in output.iter_mut().zip(input) {
        *destination = source
            .pcu_checked_to_f32_with_policy(PcuFloatUnderflowPolicy::IeeeAfterRounding)
            .map_err(|fault| format!("checked narrowing oracle failed: {fault:?}"))?;
    }
    Ok(())
}
fn narrow_fault_cases() -> Vec<(f64, PcuExecutionFaultKind)> {
    vec![
        (f64::INFINITY, PcuExecutionFaultKind::InvalidFloatingOperand),
        (f64::MAX, PcuExecutionFaultKind::ArithmeticOverflow),
        (
            f64::from_bits(1),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
    ]
}
fn wide_fill(input: &mut [f32], seed: u64) {
    for (index, value) in input.iter_mut().enumerate() {
        let bits = 0x3f00_0000_u32
            | (u32::try_from(index).unwrap_or(u32::MAX).wrapping_add(
                u32::try_from(seed & u64::from(u32::MAX)).expect("masked seed fits"),
            ) & 0x007f_ffff);
        *value = f32::from_bits(bits);
    }
}
fn wide_oracle(input: &[f32], output: &mut [f64]) -> Result<(), Box<dyn Error>> {
    for (destination, source) in output.iter_mut().zip(input) {
        *destination = source
            .pcu_checked_to_f64()
            .map_err(|fault| format!("checked widening oracle failed: {fault:?}"))?;
    }
    Ok(())
}
fn wide_fault_cases() -> Vec<(f32, PcuExecutionFaultKind)> {
    vec![
        (f32::INFINITY, PcuExecutionFaultKind::InvalidFloatingOperand),
        (f32::NAN, PcuExecutionFaultKind::InvalidFloatingOperand),
    ]
}

// Keep concrete element types visible to `#[pcu]`: the macro validates literal source syntax
// before Rust's declarative profile macro can substitute associated profile types.
mod narrow_kernel {
    use fusion_pcu_macros::pcu;

    #[pcu(invocations = N)]
    pub fn convert<const N: usize>(input: &[f64], output: &mut [f32]) {
        let id = pcu::context::global_invocation_id();
        output[id] = input[id] as f32;
    }

    pub const fn conversion_ir_bindings() -> [fusion_pcu::PcuBinding<'static>; 2] {
        [
            fusion_pcu::PcuBinding::scalar::<f64>(
                Some("input"),
                0,
                0,
                fusion_pcu::PcuBindingStorageClass::Storage,
                fusion_pcu::PcuBindingAccess::ReadOnly,
            ),
            fusion_pcu::PcuBinding::scalar::<f32>(
                Some("output"),
                0,
                1,
                fusion_pcu::PcuBindingStorageClass::Storage,
                fusion_pcu::PcuBindingAccess::WriteOnly,
            ),
        ]
    }
}

mod wide_kernel {
    use fusion_pcu_macros::pcu;

    #[pcu(invocations = N)]
    pub fn widen<const N: usize>(input: &[f32], output: &mut [f64]) {
        let id = pcu::context::global_invocation_id();
        output[id] = input[id] as f64;
    }

    pub const fn widening_ir_bindings() -> [fusion_pcu::PcuBinding<'static>; 2] {
        [
            fusion_pcu::PcuBinding::scalar::<f32>(
                Some("input"),
                0,
                0,
                fusion_pcu::PcuBindingStorageClass::Storage,
                fusion_pcu::PcuBindingAccess::ReadOnly,
            ),
            fusion_pcu::PcuBinding::scalar::<f64>(
                Some("output"),
                0,
                1,
                fusion_pcu::PcuBindingStorageClass::Storage,
                fusion_pcu::PcuBindingAccess::WriteOnly,
            ),
        ]
    }
}

checked_conversion_profile!(
    narrow,
    narrow_kernel,
    convert,
    convert_prepare,
    convert_ir,
    conversion_ir_bindings,
    f64,
    f32,
    narrow_fill,
    narrow_oracle,
    narrow_fault_cases,
    "f64-to-f32",
    "native_checked_f64_to_f32"
);
checked_conversion_profile!(
    wide,
    wide_kernel,
    widen,
    widen_prepare,
    widen_ir,
    widening_ir_bindings,
    f32,
    f64,
    wide_fill,
    wide_oracle,
    wide_fault_cases,
    "f32-to-f64",
    "native_checked_f32_to_f64"
);

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    narrow::run(criterion)?;
    wide::run(criterion)
}

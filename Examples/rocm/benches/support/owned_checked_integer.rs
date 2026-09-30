//! Matched checked integer additions with resident inputs and fresh per-call outputs.

#[path = "alloc.rs"]
#[allow(dead_code)]
mod alloc;

#[rustfmt::skip]
use std::{
    error::Error,
    mem::size_of,
    rc::Rc,
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
    PcuExecutionError,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuScalar,
    PcuTensor,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    RocmDiscovery,
    compile_hip_source,
};
use bytemuck::Pod;

const BLOCK: u32 = 256;

#[fusion_pcu::pcu]
fn source_add_u32(lhs: &[u32], rhs: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[fusion_pcu::pcu]
fn source_identity_u32(input: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::identity(input)
}

#[fusion_pcu::pcu(invocations = N)]
fn source_refresh_u32<const N: usize>(input: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[fusion_pcu::pcu]
fn source_add_i64(lhs: &[i64], rhs: &[i64]) -> Result<PcuTensor<i64>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[fusion_pcu::pcu]
fn source_identity_i64(input: &[i64]) -> Result<PcuTensor<i64>, PcuExecutionError> {
    pcu::identity(input)
}

#[fusion_pcu::pcu(invocations = N)]
fn source_refresh_i64<const N: usize>(input: &[i64], output: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

trait CheckedAdd: Pod + PcuScalar + Copy + PartialEq + std::fmt::Debug + 'static {
    fn identity(values: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn add(
        lhs: &PcuTensor<Self>,
        rhs: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn refresh(
        elements: usize,
        values: &[Self],
        resident: &mut PcuTensor<Self>,
    ) -> Result<(), PcuExecutionError>;
    fn native_source(elements: usize) -> String;
    fn fill(elements: usize, sample: usize, lhs: &mut [Self], rhs: &mut [Self]);
    fn verify(lhs: &[Self], rhs: &[Self], output: &[Self]);
    fn fault_inputs(elements: usize) -> Vec<(Vec<Self>, Vec<Self>, usize, PcuExecutionFaultKind)>;
}

impl CheckedAdd for u32 {
    fn identity(values: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        source_identity_u32(values)
    }
    fn add(
        lhs: &PcuTensor<Self>,
        rhs: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        source_add_u32(lhs, rhs)
    }
    fn refresh(
        elements: usize,
        values: &[Self],
        resident: &mut PcuTensor<Self>,
    ) -> Result<(), PcuExecutionError> {
        match elements {
            65 => source_refresh_u32::<65>(values, resident),
            1_048_576 => source_refresh_u32::<1_048_576>(values, resident),
            _ => unreachable!(),
        }
    }
    fn native_source(elements: usize) -> String {
        format!(
            "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void checked_add(const unsigned int* a,const unsigned int* b,unsigned int* o,unsigned long long* f) {{ unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>={elements}u)return; unsigned long long s=(unsigned long long)a[i]+(unsigned long long)b[i]; if(s>4294967295ull) atomicMin(f,((unsigned long long)i<<3u)|3ull); else o[i]=(unsigned int)s; }}\n"
        )
    }
    fn fill(elements: usize, sample: usize, lhs: &mut [Self], rhs: &mut [Self]) {
        for (i, (a, b)) in lhs.iter_mut().zip(rhs).enumerate() {
            *a = Self::try_from((i + sample) % 10_000).expect("bounded");
            *b = 3 + Self::try_from((i * 3 + sample) % 10_000).expect("bounded");
        }
        assert_eq!(lhs.len(), elements);
    }
    fn verify(lhs: &[Self], rhs: &[Self], output: &[Self]) {
        assert_eq!(
            output,
            lhs.iter().zip(rhs).map(|(a, b)| a + b).collect::<Vec<_>>()
        );
    }
    fn fault_inputs(elements: usize) -> Vec<(Vec<Self>, Vec<Self>, usize, PcuExecutionFaultKind)> {
        let mut a = vec![1; elements];
        let b = vec![1; elements];
        let id = elements / 2;
        a[id] = Self::MAX;
        vec![(a, b, id, PcuExecutionFaultKind::ArithmeticOverflow)]
    }
}

impl CheckedAdd for i64 {
    fn identity(values: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        source_identity_i64(values)
    }
    fn add(
        lhs: &PcuTensor<Self>,
        rhs: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        source_add_i64(lhs, rhs)
    }
    fn refresh(
        elements: usize,
        values: &[Self],
        resident: &mut PcuTensor<Self>,
    ) -> Result<(), PcuExecutionError> {
        match elements {
            65 => source_refresh_i64::<65>(values, resident),
            1_048_576 => source_refresh_i64::<1_048_576>(values, resident),
            _ => unreachable!(),
        }
    }
    fn native_source(elements: usize) -> String {
        format!(
            "#include <hip/hip_runtime.h>\n#include <limits.h>\nextern \"C\" __global__ void checked_add(const long long* a,const long long* b,long long* o,unsigned long long* f) {{ unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>={elements}u)return; long long x=a[i],y=b[i]; if(y>0 && x>LLONG_MAX-y) atomicMin(f,((unsigned long long)i<<3u)|3ull); else if(y<0 && x<LLONG_MIN-y) atomicMin(f,((unsigned long long)i<<3u)|4ull); else o[i]=x+y; }}\n"
        )
    }
    fn fill(elements: usize, sample: usize, lhs: &mut [Self], rhs: &mut [Self]) {
        for (i, (a, b)) in lhs.iter_mut().zip(rhs).enumerate() {
            *a = Self::try_from((i + sample) % 10_000).expect("bounded") - 5_000;
            *b = Self::try_from((i * 3 + sample) % 10_000).expect("bounded") - 5_000;
        }
        assert_eq!(lhs.len(), elements);
    }
    fn verify(lhs: &[Self], rhs: &[Self], output: &[Self]) {
        assert_eq!(
            output,
            lhs.iter().zip(rhs).map(|(a, b)| a + b).collect::<Vec<_>>()
        );
    }
    fn fault_inputs(elements: usize) -> Vec<(Vec<Self>, Vec<Self>, usize, PcuExecutionFaultKind)> {
        let mut a = vec![0; elements];
        let b = vec![1; elements];
        let overflow_id = elements / 3;
        a[overflow_id] = Self::MAX;
        let mut underflow_lhs = vec![0; elements];
        let underflow_rhs = vec![-1; elements];
        let underflow_id = (elements * 2) / 3;
        underflow_lhs[underflow_id] = Self::MIN;
        vec![
            (a, b, overflow_id, PcuExecutionFaultKind::ArithmeticOverflow),
            (
                underflow_lhs,
                underflow_rhs,
                underflow_id,
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ),
        ]
    }
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = crate::support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect();
    let (backend, selected) =
        crate::support::selection::open_ranked(&discovery, candidates, BLOCK)?;
    let _backend = Rc::new(backend);
    let runtime = discovery.open_device(selected.device)?;
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
        "Checked integer add benchmark device: {} ({architecture}); native includes fresh output/fault allocation, sentinel initialization, launch, wait, exact 8-byte fault readback, and both releases. Input refresh, output oracle readback, compilation, and driver allocations are excluded from Rust allocator counts.",
        selected.name
    );
    run_case::<u32>(criterion, &runtime, architecture, 65)?;
    run_case::<u32>(criterion, &runtime, architecture, 1_048_576)?;
    run_case::<i64>(criterion, &runtime, architecture, 65)?;
    run_case::<i64>(criterion, &runtime, architecture, 1_048_576)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)]
#[allow(unsafe_code)] // HIP launch calls are required for the native route in this benchmark.
fn run_case<T: CheckedAdd>(
    criterion: &mut Criterion,
    runtime: &HipRuntime,
    architecture: &str,
    elements: usize,
) -> Result<(), Box<dyn Error>> {
    let mut lhs = vec![T::zeroed(); elements];
    let mut rhs = lhs.clone();
    let mut observed = lhs.clone();
    T::fill(elements, 0, &mut lhs, &mut rhs);
    let mut source_lhs = T::identity(&lhs)?;
    let mut source_rhs = T::identity(&rhs)?;
    let source = T::add(&source_lhs, &source_rhs)?;
    source.read_into(&mut observed)?;
    T::verify(&lhs, &rhs, &observed);
    drop(source);
    let mut native_lhs = runtime.allocate(elements * size_of::<T>())?;
    let mut native_rhs = runtime.allocate(elements * size_of::<T>())?;
    native_lhs.copy_from(bytemuck::cast_slice(&lhs))?;
    native_rhs.copy_from(bytemuck::cast_slice(&rhs))?;
    let image = crate::support::cold_once("checked integer native HIP compilation", || {
        compile_hip_source(&T::native_source(elements), architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let kernel = module.function(c"checked_add")?;
    let stream = runtime.create_stream()?;
    let grid = u32::try_from(elements)?.div_ceil(BLOCK);
    let output = runtime.allocate(elements * size_of::<T>())?;
    let mut fault = runtime.allocate(8)?;
    fault.copy_from(&u64::MAX.to_le_bytes())?;
    let args = [
        HipKernelArgument::Buffer(&native_lhs),
        HipKernelArgument::Buffer(&native_rhs),
        HipKernelArgument::Buffer(&output),
        HipKernelArgument::Buffer(&fault),
    ];
    // SAFETY: These allocations satisfy the native checked-add buffer contract.
    let mut completion = unsafe { kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
    completion.wait()?;
    drop(completion);
    let mut fault_bytes = [0_u8; 8];
    fault.copy_to(&mut fault_bytes)?;
    assert_eq!(u64::from_le_bytes(fault_bytes), u64::MAX);
    output.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
    T::verify(&lhs, &rhs, &observed);
    drop((output, fault));

    let _capture = alloc::AllocationCapture::start();
    let output = T::add(&source_lhs, &source_rhs)?;
    drop(output);
    let source_allocs = alloc::AllocationCapture::finish();
    let _capture = alloc::AllocationCapture::start();
    let output = runtime.allocate(elements * size_of::<T>())?;
    let mut fault = runtime.allocate(8)?;
    fault.copy_from(&u64::MAX.to_le_bytes())?;
    let args = [
        HipKernelArgument::Buffer(&native_lhs),
        HipKernelArgument::Buffer(&native_rhs),
        HipKernelArgument::Buffer(&output),
        HipKernelArgument::Buffer(&fault),
    ];
    // SAFETY: These allocations satisfy the native checked-add buffer contract.
    let mut completion = unsafe { kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
    completion.wait()?;
    drop(completion);
    let mut status = [0_u8; 8];
    fault.copy_to(&mut status)?;
    drop((output, fault));
    let native_allocs = alloc::AllocationCapture::finish();
    println!(
        "Checked add {elements} Rust heap census: source {}/{}/{} B, native {}/{}/{} B allocations/reallocations/requested; this excludes HIP/driver allocations.",
        source_allocs.alloc_calls,
        source_allocs.realloc_calls,
        source_allocs.requested_bytes,
        native_allocs.alloc_calls,
        native_allocs.realloc_calls,
        native_allocs.requested_bytes,
    );
    verify_fault_policy::<T>(
        runtime,
        &kernel,
        &stream,
        grid,
        elements,
        &mut source_lhs,
        &mut source_rhs,
    )?;

    let mut group = criterion.benchmark_group("owned_checked_integer_add");
    group.throughput(Throughput::Elements(u64::try_from(elements)?));
    let mut sample = 1_usize;
    group.bench_function(
        BenchmarkId::new(format!("{}/source", std::any::type_name::<T>()), elements),
        |b| {
            b.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    T::fill(elements, sample, &mut lhs, &mut rhs);
                    sample = sample.wrapping_add(1);
                    T::refresh(elements, &lhs, &mut source_lhs).expect("source lhs refresh");
                    T::refresh(elements, &rhs, &mut source_rhs).expect("source rhs refresh");
                    let start = Instant::now();
                    let output = T::add(&source_lhs, &source_rhs).expect("source checked addition");
                    let operation = start.elapsed();
                    output
                        .read_into(&mut observed)
                        .expect("source oracle readback");
                    T::verify(&lhs, &rhs, &observed);
                    let release = Instant::now();
                    drop(output);
                    total += operation + release.elapsed();
                }
                total
            });
        },
    );
    group.bench_function(
        BenchmarkId::new(format!("{}/native", std::any::type_name::<T>()), elements),
        |b| {
            b.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    T::fill(elements, sample, &mut lhs, &mut rhs);
                    sample = sample.wrapping_add(1);
                    native_lhs
                        .copy_from(bytemuck::cast_slice(&lhs))
                        .expect("native lhs refresh");
                    native_rhs
                        .copy_from(bytemuck::cast_slice(&rhs))
                        .expect("native rhs refresh");
                    let start = Instant::now();
                    let output = runtime
                        .allocate(elements * size_of::<T>())
                        .expect("native output allocation");
                    let mut fault = runtime.allocate(8).expect("native fault allocation");
                    fault
                        .copy_from(&u64::MAX.to_le_bytes())
                        .expect("native fault initialization");
                    let args = [
                        HipKernelArgument::Buffer(&native_lhs),
                        HipKernelArgument::Buffer(&native_rhs),
                        HipKernelArgument::Buffer(&output),
                        HipKernelArgument::Buffer(&fault),
                    ];
                    // SAFETY: Each resident input/output covers `elements` values of T; fault is 8 bytes.
                    let mut completion =
                        unsafe { kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }
                            .expect("native checked add launch");
                    completion.wait().expect("native checked add completion");
                    drop(completion);
                    let mut word = [0_u8; 8];
                    fault.copy_to(&mut word).expect("native fault readback");
                    assert_eq!(
                        u64::from_le_bytes(word),
                        u64::MAX,
                        "valid benchmark inputs faulted"
                    );
                    drop(fault);
                    let operation = start.elapsed();
                    output
                        .copy_to(bytemuck::cast_slice_mut(&mut observed))
                        .expect("native output readback");
                    T::verify(&lhs, &rhs, &observed);
                    let release = Instant::now();
                    drop(output);
                    total += operation + release.elapsed();
                }
                total
            });
        },
    );
    group.finish();
    Ok(())
}

#[allow(unsafe_code)] // HIP launch calls exercise the same checked-add kernel as the source route.
#[allow(clippy::too_many_arguments)]
fn verify_fault_policy<T: CheckedAdd>(
    runtime: &HipRuntime,
    kernel: &HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    grid: u32,
    elements: usize,
    source_lhs: &mut PcuTensor<T>,
    source_rhs: &mut PcuTensor<T>,
) -> Result<(), Box<dyn Error>> {
    for (lhs, rhs, id, kind) in T::fault_inputs(elements) {
        T::refresh(elements, &lhs, source_lhs)?;
        T::refresh(elements, &rhs, source_rhs)?;
        let expected = PcuExecutionFault {
            recovered: false,
            invocation_id: u64::try_from(id)?,
            kind,
        };
        let source_fault = match T::add(source_lhs, source_rhs) {
            Err(PcuExecutionError::ArithmeticFault(fault)) => fault,
            other => {
                return Err(
                    format!("source checked addition did not return {kind:?}: {other:?}").into(),
                );
            }
        };
        let mut a = runtime.allocate(elements * size_of::<T>())?;
        let mut b = runtime.allocate(elements * size_of::<T>())?;
        let output = runtime.allocate(elements * size_of::<T>())?;
        let mut fault = runtime.allocate(8)?;
        a.copy_from(bytemuck::cast_slice(&lhs))?;
        b.copy_from(bytemuck::cast_slice(&rhs))?;
        fault.copy_from(&u64::MAX.to_le_bytes())?;
        let args = [
            HipKernelArgument::Buffer(&a),
            HipKernelArgument::Buffer(&b),
            HipKernelArgument::Buffer(&output),
            HipKernelArgument::Buffer(&fault),
        ];
        // SAFETY: Input/output extents match T; the fault argument is exactly one u64.
        let mut completion =
            unsafe { kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
        completion.wait()?;
        drop(completion);
        let mut bytes = [0_u8; 8];
        fault.copy_to(&mut bytes)?;
        let word = u64::from_le_bytes(bytes);
        let native_fault = match word {
            u64::MAX => return Err(format!("native {kind:?} fixture did not fault").into()),
            word => PcuExecutionFault {
                recovered: false,
                invocation_id: word >> 3,
                kind: match word & 7 {
                    3 => PcuExecutionFaultKind::ArithmeticOverflow,
                    4 => PcuExecutionFaultKind::ArithmeticUnderflow,
                    _ => return Err(format!("invalid native fault word {word}").into()),
                },
            },
        };
        assert_eq!(source_fault, expected);
        assert_eq!(native_fault, source_fault);
        println!(
            "Checked integer {kind:?} verified: source and HIP publish {native_fault:?}; terminal word is {word:#018x}."
        );
    }
    Ok(())
}

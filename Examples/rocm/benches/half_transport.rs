//! Bit-preserving f16/bf16 storage transport against independent unsigned-short HIP copies.

#[path = "support/dispatch.rs"]
#[allow(dead_code)]
mod dispatch_support;
#[allow(dead_code)]
mod support;

#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
    num::NonZeroU32,
    time::Duration,
};

#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBf16Bits,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuF16Bits,
    PcuInvocationShape,
    PcuValueType,
    PcuOwnedCompletion,
    PcuScalar,
};
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use fusion_pcu_rocm::{
    compile_hip_source,
    HipKernelArgument,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
};
use support::selection;
const BLOCK_SIZE: u32 = 64;

const INPUT: PcuBindingRef = PcuBindingRef::new(0, 0);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 1);

#[pcu(invocations = N)]
fn generic_identity<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[pcu(invocations = I)]
fn generic_grid_identity<T: PcuScalar, const I: usize, const N: usize>(
    input: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}

fn pattern(n: usize) -> Vec<u16> {
    const EDGES: [u16; 16] = [
        0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x7bff, 0x7c00, 0xfc00, 0x7e00, 0x7fff, 0xffff,
        0x7f80, 0xff80, 0x0080, 0x3555, 0xaaaa,
    ];
    let mut values = (0..n)
        .map(|i| {
            u16::try_from(i % 65_536)
                .expect("bounded pattern index")
                .wrapping_mul(0x9e37)
                .wrapping_add(0x3555)
        })
        .collect::<Vec<_>>();
    for (destination, edge) in values.iter_mut().zip(EDGES) {
        *destination = edge;
    }
    values
}

fn run(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect::<Vec<_>>();
    let (backend, selected) = selection::open_ranked(&discovery, candidates, BLOCK_SIZE)?;
    let arch = selected
        .architecture
        .ok_or("selected device has no HIP architecture")?;
    let runtime = discovery.open_device(selected.device)?;
    println!(
        "half identity transport device: {} ({arch})",
        discovery.device_info(selected.device)?.name
    );
    for (label, ty, type_name) in [
        ("f16", PcuValueType::f16(), "f16"),
        ("bf16", PcuValueType::bf16(), "bf16"),
    ] {
        for n in [65_usize, 1 << 20] {
            run_case(
                c,
                &backend,
                &runtime,
                &arch,
                label,
                ty,
                type_name,
                u32::try_from(n)?,
                n,
                false,
            )?;
        }
        run_case(
            c, &backend, &runtime, &arch, label, ty, type_name, 2048, 250, true,
        )?;
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::significant_drop_tightening
)]
fn run_case(
    c: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &fusion_pcu_rocm::HipRuntime,
    arch: &str,
    label: &str,
    ty: PcuValueType,
    type_name: &str,
    extent: u32,
    invocations: usize,
    grid_stride: bool,
) -> Result<(), Box<dyn Error>> {
    let input = pattern(usize::try_from(extent)?);
    let bytes = input
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let mut pcu_in = backend.allocate(bytes.len())?;
    let pcu_out = backend.allocate(bytes.len())?;
    pcu_in.copy_from(&bytes)?;
    let mut hip_in = runtime.allocate(bytes.len())?;
    let hip_out = runtime.allocate(bytes.len())?;
    hip_in.copy_from(&bytes)?;
    let submitted = u32::try_from(invocations)?;
    let bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            ty,
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
            ty,
        ),
    ];
    let builder = match (ty, invocations, grid_stride) {
        (value, 65, false) if value == PcuValueType::f16() => {
            generic_identity_ir::<PcuF16Bits, 65>(&bindings)
        }
        (value, 1_048_576, false) if value == PcuValueType::f16() => {
            generic_identity_ir::<PcuF16Bits, 1_048_576>(&bindings)
        }
        (value, 65, false) if value == PcuValueType::bf16() => {
            generic_identity_ir::<PcuBf16Bits, 65>(&bindings)
        }
        (value, 1_048_576, false) if value == PcuValueType::bf16() => {
            generic_identity_ir::<PcuBf16Bits, 1_048_576>(&bindings)
        }
        (value, 250, true) if value == PcuValueType::f16() => {
            generic_grid_identity_ir::<PcuF16Bits, 250, 2048>(&bindings)
        }
        (value, 250, true) if value == PcuValueType::bf16() => {
            generic_grid_identity_ir::<PcuBf16Bits, 250, 2048>(&bindings)
        }
        _ => return Err("unsupported generic identity benchmark specialization".into()),
    }
    .map_err(|error| format!("generic identity: {error:?}"))?;
    let specialized = builder.ir();
    let shape =
        PcuInvocationShape::invocations(NonZeroU32::new(submitted).ok_or("zero invocation count")?);
    let prepared = backend.prepare_dispatch(fusion_pcu::PcuDispatchSubmission {
        kernel: &specialized,
        shape,
    })?;
    let source = "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void direct_copy(const unsigned short* in, unsigned short* out, unsigned n) { unsigned id=blockIdx.x*blockDim.x+threadIdx.x; if(id<n) out[id]=in[id]; }\nextern \"C\" __global__ void grid_copy(const unsigned short* in, unsigned short* out, unsigned n, unsigned stride) { unsigned id=blockIdx.x*blockDim.x+threadIdx.x; if(id<stride) for(;id<n;id+=stride) out[id]=in[id]; }\n";
    let module = runtime.load_module(&compile_hip_source(source, arch)?)?;
    let function = module.function(if grid_stride {
        c"grid_copy"
    } else {
        c"direct_copy"
    })?;
    let stream = runtime.create_stream()?;
    let n_bytes = extent.to_ne_bytes();
    let stride_bytes = submitted.to_ne_bytes();
    let direct_args = [
        HipKernelArgument::Buffer(&hip_in),
        HipKernelArgument::Buffer(&hip_out),
        HipKernelArgument::Bytes(&n_bytes),
    ];
    let grid_args = [
        HipKernelArgument::Buffer(&hip_in),
        HipKernelArgument::Buffer(&hip_out),
        HipKernelArgument::Bytes(&n_bytes),
        HipKernelArgument::Bytes(&stride_bytes),
    ];
    let args = if grid_stride {
        &grid_args[..]
    } else {
        &direct_args[..]
    };
    let grid = if grid_stride {
        submitted.div_ceil(BLOCK_SIZE)
    } else {
        extent.div_ceil(BLOCK_SIZE)
    };
    let refs = [
        backend.binding(
            INPUT,
            PcuBindingAccess::ReadOnly,
            fusion_pcu::PcuBindingType::Value(ty),
            pcu_in.clone(),
        )?,
        backend.binding(
            OUTPUT,
            PcuBindingAccess::ReadWrite,
            fusion_pcu::PcuBindingType::Value(ty),
            pcu_out.clone(),
        )?,
    ];
    let mut completion = prepared.submit(&refs)?;
    if completion.wait()? != fusion_pcu::PcuCompletionOutcome::Succeeded {
        return Err("PCU half identity preflight failed".into());
    }
    dispatch_support::run_direct(&function, &stream, args, grid)?;
    let mut pcu_result = vec![0_u8; bytes.len()];
    let mut hip_result = vec![0_u8; bytes.len()];
    pcu_out.copy_to(&mut pcu_result)?;
    hip_out.copy_to(&mut hip_result)?;
    if pcu_result != bytes || hip_result != bytes || pcu_result != hip_result {
        return Err(format!(
            "{label} copy changed input bits (extent={extent}, invocations={invocations})"
        )
        .into());
    }
    println!(
        "{label} identity copy verified bit-for-bit: extent={extent}, invocations={invocations}, grid-stride={grid_stride}"
    );
    let mut group = c.benchmark_group(format!(
        "{type_name}-identity-copy-{}",
        if grid_stride { "grid-stride" } else { "direct" }
    ));
    group.throughput(Throughput::Elements(u64::from(extent)));
    for (name, is_pcu) in [("Prepared PCU", true), ("Direct HIP", false)] {
        group.bench_function(
            BenchmarkId::new(name, format!("n{extent}-i{invocations}")),
            |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for iteration in 0..iterations {
                        let (pcu, native) = if iteration.is_multiple_of(2) {
                            (
                                run_pcu(&prepared, &refs),
                                run_native(&function, &stream, args, grid),
                            )
                        } else {
                            let native = run_native(&function, &stream, args, grid);
                            let pcu = run_pcu(&prepared, &refs);
                            (pcu, native)
                        };
                        elapsed += if is_pcu { pcu } else { native };
                    }
                    black_box(elapsed)
                });
            },
        );
    }
    group.finish();
    Ok(())
}

fn run_pcu(
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
    refs: &[fusion_pcu::PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
) -> Duration {
    let start = std::time::Instant::now();
    let mut completion = prepared.submit(refs).expect("PCU submit");
    assert_eq!(
        completion.wait().expect("PCU completion"),
        fusion_pcu::PcuCompletionOutcome::Succeeded
    );
    start.elapsed()
}

fn run_native(
    function: &fusion_pcu_rocm::HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    args: &[HipKernelArgument<'_>],
    grid: u32,
) -> Duration {
    dispatch_support::run_direct(function, stream, args, grid)
        .expect("HIP launch")
        .total
}

fn benchmarks(c: &mut Criterion) {
    run(c).expect("half transport benchmark failed");
}
criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = benchmarks
}
criterion_main!(benches);

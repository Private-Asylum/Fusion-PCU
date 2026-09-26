//! Compare wrapping i16 PCU Add with independent native HIP.

#[path = "support/dispatch.rs"]
#[allow(dead_code)]
mod dispatch_support;
mod support;

use std::{
    error::Error,
    hint::black_box,
    num::NonZeroU32,
    time::Duration,
};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};
use fusion_pcu::{
    PcuDispatchSubmission,
    PcuInvocationShape,
    PcuValueType,
};
use fusion_pcu_macros::pcu;
use fusion_pcu_rocm::{
    compile_hip_source,
    HipKernelArgument,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
};
use dispatch_support::{
    print_samples,
    run_direct,
    run_prepared_typed_binary,
    BLOCK_SIZE,
};

#[pcu(invocations = N)]
fn wrapping_add_i16<const N: usize>(left: &[i16], right: &[i16], output: &mut [i16]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id].wrapping_add(right[id]);
}

#[pcu(invocations = 64)]
fn wrapping_add_grid_i16<const N: usize>(left: &[i16], right: &[i16], output: &mut [i16]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id].wrapping_add(right[id]);
        id += stride;
    }
}

fn encode(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("i16 wrapping Add benchmark failed");
}

fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect::<Vec<_>>();
    let (backend, selected) = support::selection::open_ranked(&discovery, candidates, BLOCK_SIZE)?;
    let architecture = selected
        .architecture
        .ok_or("selected device has no HIP architecture")?;
    let runtime = discovery.open_device(selected.device)?;
    println!(
        "i16 wrapping Add device: {} ({architecture})",
        discovery.device_info(selected.device)?.name
    );
    run_case::<65>(criterion, &backend, &runtime, &architecture)?;
    run_case::<{ 1 << 20 }>(criterion, &backend, &runtime, &architecture)?;
    Ok(())
}

#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)]
fn run_case<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &fusion_pcu_rocm::HipRuntime,
    architecture: &str,
) -> Result<(), Box<dyn Error>> {
    let left = (0..N)
        .map(|i| match i {
            0 => i16::MAX,
            1 => i16::MIN,
            _ => i16::try_from(i % (usize::from(u16::MAX) / 2 + 1))
                .expect("bounded i16 input fits")
                .wrapping_mul(251),
        })
        .collect::<Vec<_>>();
    let right = (0..N)
        .map(|i| match i {
            0 => 1,
            1 => -1,
            _ => i16::try_from(i % (usize::from(u16::MAX) / 2 + 1))
                .expect("bounded i16 input fits")
                .wrapping_mul(17)
                .wrapping_add(3),
        })
        .collect::<Vec<_>>();
    let expected = left
        .iter()
        .zip(&right)
        .map(|(a, b)| a.wrapping_add(*b))
        .collect::<Vec<_>>();
    let left_bytes = encode(&left);
    let right_bytes = encode(&right);
    let mut pcu_left = backend.allocate(left_bytes.len())?;
    let mut pcu_right = backend.allocate(right_bytes.len())?;
    let pcu_output = backend.allocate(left_bytes.len())?;
    pcu_left.copy_from(&left_bytes)?;
    pcu_right.copy_from(&right_bytes)?;
    let mut native_left = runtime.allocate(left_bytes.len())?;
    let mut native_right = runtime.allocate(right_bytes.len())?;
    let native_output = runtime.allocate(left_bytes.len())?;
    native_left.copy_from(&left_bytes)?;
    native_right.copy_from(&right_bytes)?;

    let bindings = wrapping_add_i16_bindings();
    let builder = wrapping_add_i16::<N>(&bindings)?;
    let kernel = builder.ir();
    let invocations = u32::try_from(N)?;
    let prepared = support::cold_once("PCU i16 cold prepare", || {
        backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(invocations).expect("nonzero")),
        })
    })?;

    let grid_bindings = wrapping_add_grid_i16_bindings();
    let grid_builder = wrapping_add_grid_i16::<N>(&grid_bindings)?;
    let grid_kernel = grid_builder.ir();
    let grid_prepared = support::cold_once("PCU i16 grid-stride cold prepare", || {
        backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &grid_kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(BLOCK_SIZE).expect("nonzero")),
        })
    })?;

    let source = String::from(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_i16_direct(const unsigned short* left, const unsigned short* right, unsigned short* output, unsigned int n) { unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < n) output[id] = static_cast<unsigned short>((static_cast<unsigned int>(left[id]) + static_cast<unsigned int>(right[id])) & 0xffffu); }\nextern \"C\" __global__ void native_i16_grid(const unsigned short* left, const unsigned short* right, unsigned short* output, unsigned int n) { unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; unsigned int stride = gridDim.x * blockDim.x; for (; id < n; id += stride) output[id] = static_cast<unsigned short>((static_cast<unsigned int>(left[id]) + static_cast<unsigned int>(right[id])) & 0xffffu); }\n",
    );
    let image = compile_hip_source(&source, architecture)?;
    let module = runtime.load_module(&image)?;
    let direct = module.function(c"native_i16_direct")?;
    let grid_function = module.function(c"native_i16_grid")?;
    let stream = runtime.create_stream()?;
    let n = invocations;
    let n_bytes = n.to_ne_bytes();
    let direct_args = [
        HipKernelArgument::Buffer(&native_left),
        HipKernelArgument::Buffer(&native_right),
        HipKernelArgument::Buffer(&native_output),
        HipKernelArgument::Bytes(&n_bytes),
    ];
    let grid_args = [
        HipKernelArgument::Buffer(&native_left),
        HipKernelArgument::Buffer(&native_right),
        HipKernelArgument::Buffer(&native_output),
        HipKernelArgument::Bytes(&n_bytes),
    ];
    let grid = invocations.div_ceil(BLOCK_SIZE);

    // Direct and grid-stride PCU plus independent HIP outputs must match every value.
    run_prepared_typed_binary(
        backend,
        &prepared,
        PcuValueType::i16(),
        &pcu_left,
        &pcu_right,
        &pcu_output,
    )?;
    run_direct(&direct, &stream, &direct_args, grid)?;
    verify(&pcu_output, &native_output, &expected, N, "direct")?;
    run_prepared_typed_binary(
        backend,
        &grid_prepared,
        PcuValueType::i16(),
        &pcu_left,
        &pcu_right,
        &pcu_output,
    )?;
    run_direct(&grid_function, &stream, &grid_args, grid)?;
    verify(&pcu_output, &native_output, &expected, N, "grid-stride")?;
    println!("i16 wrapping Add direct and grid-stride verified for {N} values, including overflow");

    let mut group = criterion.benchmark_group("i16-wrapping-add");
    group.throughput(Throughput::Elements(u64::try_from(N)?));
    for (label, pcu_target) in [("Prepared PCU", true), ("Direct HIP", false)] {
        let mut observed = Vec::new();
        group.bench_function(BenchmarkId::new(label, N), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for iteration in 0..iterations {
                    let (pcu, native) = if iteration.is_multiple_of(2) {
                        let pcu = run_prepared_typed_binary(
                            backend,
                            &prepared,
                            PcuValueType::i16(),
                            &pcu_left,
                            &pcu_right,
                            &pcu_output,
                        )
                        .expect("PCU i16 launch failed");
                        let native = run_direct(&direct, &stream, &direct_args, grid)
                            .expect("native i16 launch failed");
                        (pcu, native)
                    } else {
                        let native = run_direct(&direct, &stream, &direct_args, grid)
                            .expect("native i16 launch failed");
                        let pcu = run_prepared_typed_binary(
                            backend,
                            &prepared,
                            PcuValueType::i16(),
                            &pcu_left,
                            &pcu_right,
                            &pcu_output,
                        )
                        .expect("PCU i16 launch failed");
                        (pcu, native)
                    };
                    elapsed += if pcu_target { pcu.total } else { native.total };
                    observed.push(black_box(if pcu_target { pcu } else { native }));
                }
                elapsed
            });
        });
        print_samples(label, &observed);
    }
    group.finish();
    Ok(())
}

fn verify(
    pcu_output: &fusion_pcu_rocm::DeviceBuffer,
    native_output: &fusion_pcu_rocm::DeviceBuffer,
    expected: &[i16],
    n: usize,
    phase: &str,
) -> Result<(), Box<dyn Error>> {
    let mut pcu_bytes = vec![0; n * 2];
    let mut native_bytes = vec![0; n * 2];
    pcu_output.copy_to(&mut pcu_bytes)?;
    native_output.copy_to(&mut native_bytes)?;
    let expected_bytes = encode(expected);
    if pcu_bytes != expected_bytes || native_bytes != expected_bytes {
        return Err(format!("i16 {phase} PCU/native output mismatch").into());
    }
    Ok(())
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);

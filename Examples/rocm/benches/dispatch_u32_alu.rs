//! Compare explicit wrapping u32 PCU arithmetic with an independent HIP kernel.

#[path = "support/dispatch.rs"]
#[allow(dead_code)] // Shared timing support also serves other dispatch benchmarks.
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
};
use fusion_pcu_macros::pcu;
use fusion_pcu_rocm::{
    compile_hip_source,
    HipKernelArgument,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
};

use dispatch_support::{
    encode_u32,
    print_samples,
    run_direct,
    run_prepared_u32_binary,
    verify_u32_copy,
    BLOCK_SIZE,
};

#[pcu(invocations = N)]
fn wrapping_add_u32<const N: usize>(left: &[u32], right: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id].wrapping_add(right[id]);
}

#[pcu(invocations = N)]
fn wrapping_add_mul_u32<const N: usize>(left: &[u32], right: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id].wrapping_add(right[id]).wrapping_mul(right[id]);
}

#[pcu(invocations = 65)]
fn wrapping_sub_u32(left: &[u32], right: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id].wrapping_sub(right[id]);
}

#[pcu(invocations = 17)]
fn wrapping_mul_grid_u32<const N: usize>(left: &[u32], right: &[u32], output: &mut [u32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id].wrapping_mul(right[id]);
        id += stride;
    }
}

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("u32 wrapping arithmetic benchmark setup failed");
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
    let direct_runtime = discovery.open_device(selected.device)?;
    println!(
        "u32 wrapping Add device: {} ({architecture})",
        discovery.device_info(selected.device)?.name
    );
    verify_other_u32_operations(&backend)?;
    run_case::<65>(criterion, &backend, &direct_runtime, &architecture)?;
    run_case::<{ 1 << 20 }>(criterion, &backend, &direct_runtime, &architecture)?;
    run_chain_case::<65>(criterion, &backend, &direct_runtime, &architecture)?;
    run_chain_case::<{ 1 << 20 }>(criterion, &backend, &direct_runtime, &architecture)?;
    Ok(())
}

#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // Keep native/PCU setup, timed pairing, and correctness proof aligned.
fn run_chain_case<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    direct_runtime: &fusion_pcu_rocm::HipRuntime,
    architecture: &str,
) -> Result<(), Box<dyn Error>> {
    let left = (0..N)
        .map(|index| u32::MAX.wrapping_sub(u32::try_from(index).expect("benchmark index fits u32")))
        .collect::<Vec<_>>();
    let right = (0..N)
        .map(|index| {
            u32::try_from(index)
                .expect("benchmark index fits u32")
                .wrapping_mul(17)
                .wrapping_add(3)
        })
        .collect::<Vec<_>>();
    let expected = left
        .iter()
        .zip(&right)
        .map(|(a, b)| a.wrapping_add(*b).wrapping_mul(*b))
        .collect::<Vec<_>>();
    let left_bytes = encode_u32(&left);
    let right_bytes = encode_u32(&right);
    let mut pcu_left = backend.allocate(left_bytes.len())?;
    let mut pcu_right = backend.allocate(right_bytes.len())?;
    let pcu_output = backend.allocate(left_bytes.len())?;
    pcu_left.copy_from(&left_bytes)?;
    pcu_right.copy_from(&right_bytes)?;
    let mut native_left = direct_runtime.allocate(left_bytes.len())?;
    let mut native_right = direct_runtime.allocate(right_bytes.len())?;
    let native_output = direct_runtime.allocate(left_bytes.len())?;
    native_left.copy_from(&left_bytes)?;
    native_right.copy_from(&right_bytes)?;

    let bindings = wrapping_add_mul_u32_bindings();
    let builder = wrapping_add_mul_u32::<N>(&bindings)?;
    let kernel = builder.ir();
    let invocations = u32::try_from(N)?;
    let prepared = support::cold_once("PCU chained u32 cold prepare", || {
        backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(invocations).expect("nonzero")),
        })
    })?;
    let source = format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_u32_add_mul(const unsigned int* left, const unsigned int* right, unsigned int* output) {{\n    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;\n    if (id < {N}u) output[id] = (left[id] + right[id]) * right[id];\n}}\n"
    );
    let image = compile_hip_source(&source, architecture)?;
    let native_module = direct_runtime.load_module(&image)?;
    let native_function = native_module.function(c"native_u32_add_mul")?;
    let native_stream = direct_runtime.create_stream()?;
    let native_arguments = [
        HipKernelArgument::Buffer(&native_left),
        HipKernelArgument::Buffer(&native_right),
        HipKernelArgument::Buffer(&native_output),
    ];
    let grid = invocations.div_ceil(BLOCK_SIZE);

    // Prove both generated kernels before recording benchmark samples.
    run_prepared_u32_binary(backend, &prepared, &pcu_left, &pcu_right, &pcu_output)?;
    run_direct(&native_function, &native_stream, &native_arguments, grid)?;
    let mut pcu_bytes = vec![0; left_bytes.len()];
    let mut native_bytes = vec![0; left_bytes.len()];
    pcu_output.copy_to(&mut pcu_bytes)?;
    native_output.copy_to(&mut native_bytes)?;
    verify_u32_copy("PCU chained wrapping Add/Mul", &pcu_bytes, &expected)?;
    verify_u32_copy("native chained wrapping Add/Mul", &native_bytes, &expected)?;
    println!("u32 wrapping Add/Mul chain verified for {N} values, including overflow");

    let mut group = criterion.benchmark_group("u32-wrapping-add-mul-chain");
    group.throughput(Throughput::Elements(N as u64));
    for (label, pcu_target) in [("Prepared PCU", true), ("Direct HIP", false)] {
        let mut observed = Vec::new();
        group.bench_function(BenchmarkId::new(label, N), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for iteration in 0..iterations {
                    let (pcu, native) = if iteration.is_multiple_of(2) {
                        let pcu = run_prepared_u32_binary(
                            backend,
                            &prepared,
                            &pcu_left,
                            &pcu_right,
                            &pcu_output,
                        )
                        .expect("PCU chained u32 operation failed");
                        let native =
                            run_direct(&native_function, &native_stream, &native_arguments, grid)
                                .expect("native chained u32 operation failed");
                        (pcu, native)
                    } else {
                        let native =
                            run_direct(&native_function, &native_stream, &native_arguments, grid)
                                .expect("native chained u32 operation failed");
                        let pcu = run_prepared_u32_binary(
                            backend,
                            &prepared,
                            &pcu_left,
                            &pcu_right,
                            &pcu_output,
                        )
                        .expect("PCU chained u32 operation failed");
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

fn verify_other_u32_operations(backend: &RocmOwnedDispatchBackend) -> Result<(), Box<dyn Error>> {
    let left = (0..65_u32)
        .map(|value| u32::MAX - value)
        .collect::<Vec<_>>();
    let right = (0..65_u32).map(|value| value + 3).collect::<Vec<_>>();
    let mut device_left = backend.allocate(left.len() * 4)?;
    let mut device_right = backend.allocate(right.len() * 4)?;
    let device_output = backend.allocate(left.len() * 4)?;
    device_left.copy_from(&encode_u32(&left))?;
    device_right.copy_from(&encode_u32(&right))?;
    {
        let bindings = wrapping_sub_u32_bindings();
        let builder = wrapping_sub_u32(&bindings)?;
        let kernel = builder.ir();
        let prepared = backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(65).expect("nonzero")),
        })?;
        run_prepared_u32_binary(
            backend,
            &prepared,
            &device_left,
            &device_right,
            &device_output,
        )?;
        let mut bytes = vec![0; left.len() * 4];
        device_output.copy_to(&mut bytes)?;
        let expected = left
            .iter()
            .zip(&right)
            .map(|(a, b)| a.wrapping_sub(*b))
            .collect::<Vec<_>>();
        verify_u32_copy("wrapping Sub", &bytes, &expected)?;
    }
    {
        let bindings = wrapping_mul_grid_u32_bindings();
        let builder = wrapping_mul_grid_u32::<65>(&bindings)?;
        let kernel = builder.ir();
        let prepared = backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(17).expect("nonzero")),
        })?;
        run_prepared_u32_binary(
            backend,
            &prepared,
            &device_left,
            &device_right,
            &device_output,
        )?;
        let mut bytes = vec![0; left.len() * 4];
        device_output.copy_to(&mut bytes)?;
        let expected = left
            .iter()
            .zip(&right)
            .map(|(a, b)| a.wrapping_mul(*b))
            .collect::<Vec<_>>();
        verify_u32_copy("grid-stride wrapping Mul", &bytes, &expected)?;
    }
    println!("u32 wrapping Sub and grid-stride Mul verified for 65 values");
    Ok(())
}

#[allow(clippy::too_many_lines)] // Keep paired resources, submissions, and verification in one benchmark case.
fn run_case<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    direct_runtime: &fusion_pcu_rocm::HipRuntime,
    architecture: &str,
) -> Result<(), Box<dyn Error>> {
    let left = (0..N)
        .map(|index| u32::MAX.wrapping_sub(u32::try_from(index).expect("benchmark index fits u32")))
        .collect::<Vec<_>>();
    let right = (0..N)
        .map(|index| {
            u32::try_from(index)
                .expect("benchmark index fits u32")
                .wrapping_mul(17)
                .wrapping_add(3)
        })
        .collect::<Vec<_>>();
    let expected = left
        .iter()
        .zip(&right)
        .map(|(a, b)| a.wrapping_add(*b))
        .collect::<Vec<_>>();
    let left_bytes = encode_u32(&left);
    let right_bytes = encode_u32(&right);
    let mut pcu_left = backend.allocate(left_bytes.len())?;
    let mut pcu_right = backend.allocate(right_bytes.len())?;
    let pcu_output = backend.allocate(left_bytes.len())?;
    pcu_left.copy_from(&left_bytes)?;
    pcu_right.copy_from(&right_bytes)?;
    let mut native_left = direct_runtime.allocate(left_bytes.len())?;
    let mut native_right = direct_runtime.allocate(right_bytes.len())?;
    let native_output = direct_runtime.allocate(left_bytes.len())?;
    native_left.copy_from(&left_bytes)?;
    native_right.copy_from(&right_bytes)?;

    let bindings = wrapping_add_u32_bindings();
    let builder = wrapping_add_u32::<N>(&bindings)?;
    let kernel = builder.ir();
    let invocations = u32::try_from(N)?;
    let prepared = support::cold_once("PCU cold prepare", || {
        backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(invocations).expect("nonzero")),
        })
    })?;
    let source = format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_u32_add(const unsigned int* left, const unsigned int* right, unsigned int* output) {{\n    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;\n    if (id < {N}u) output[id] = left[id] + right[id];\n}}\n"
    );
    let image = compile_hip_source(&source, architecture)?;
    let native_module = direct_runtime.load_module(&image)?;
    let native_function = native_module.function(c"native_u32_add")?;
    let native_stream = direct_runtime.create_stream()?;
    let native_arguments = [
        HipKernelArgument::Buffer(&native_left),
        HipKernelArgument::Buffer(&native_right),
        HipKernelArgument::Buffer(&native_output),
    ];
    let grid = invocations.div_ceil(BLOCK_SIZE);

    {
        let mut group = criterion.benchmark_group("u32-wrapping-add");
        group.throughput(Throughput::Elements(N as u64));
        for (label, pcu_target) in [("Prepared PCU", true), ("Direct HIP", false)] {
            let mut observed = Vec::new();
            group.bench_function(BenchmarkId::new(label, N), |bencher| {
                bencher.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for iteration in 0..iterations {
                        let (pcu, native) = if iteration.is_multiple_of(2) {
                            let pcu = run_prepared_u32_binary(
                                backend,
                                &prepared,
                                &pcu_left,
                                &pcu_right,
                                &pcu_output,
                            )
                            .expect("PCU u32 Add failed");
                            let native = run_direct(
                                &native_function,
                                &native_stream,
                                &native_arguments,
                                grid,
                            )
                            .expect("native u32 Add failed");
                            (pcu, native)
                        } else {
                            let native = run_direct(
                                &native_function,
                                &native_stream,
                                &native_arguments,
                                grid,
                            )
                            .expect("native u32 Add failed");
                            let pcu = run_prepared_u32_binary(
                                backend,
                                &prepared,
                                &pcu_left,
                                &pcu_right,
                                &pcu_output,
                            )
                            .expect("PCU u32 Add failed");
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
    }

    let mut pcu_bytes = vec![0; left_bytes.len()];
    let mut native_bytes = vec![0; left_bytes.len()];
    pcu_output.copy_to(&mut pcu_bytes)?;
    native_output.copy_to(&mut native_bytes)?;
    verify_u32_copy("PCU wrapping Add", &pcu_bytes, &expected)?;
    verify_u32_copy("native wrapping Add", &native_bytes, &expected)?;
    println!("u32 wrapping Add verified for {N} values, including overflow");
    Ok(())
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = bench
}
criterion_main!(benches);

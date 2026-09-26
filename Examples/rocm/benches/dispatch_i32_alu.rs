//! Compare a chained wrapping i32 PCU kernel with independent native HIP.

#[path = "support/dispatch.rs"]
#[allow(dead_code)] // Shared timing support also serves the other dispatch benchmarks.
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
fn wrapping_add_mul_i32<const N: usize>(left: &[i32], right: &[i32], output: &mut [i32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id].wrapping_add(right[id]).wrapping_mul(right[id]);
}

fn encode(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("paired i32 benchmark failed");
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
    println!(
        "i32 wrapping Add/Mul device: {} ({architecture})",
        discovery.device_info(selected.device)?.name
    );
    let direct_runtime = discovery.open_device(selected.device)?;
    run_case::<65>(criterion, &backend, &direct_runtime, &architecture)?;
    run_case::<{ 1 << 20 }>(criterion, &backend, &direct_runtime, &architecture)?;
    Ok(())
}

#[allow(clippy::too_many_lines)] // Keep matching resources, launches, and verification together.
fn run_case<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &fusion_pcu_rocm::HipRuntime,
    architecture: &str,
) -> Result<(), Box<dyn Error>> {
    let left = (0..N)
        .map(|index| i32::try_from(index).map(|value| i32::MAX.wrapping_sub(value)))
        .collect::<Result<Vec<_>, _>>()?;
    let right = (0..N)
        .map(|index| i32::try_from(index).map(|value| value.wrapping_mul(17).wrapping_add(3)))
        .collect::<Result<Vec<_>, _>>()?;
    let expected = left
        .iter()
        .zip(&right)
        .map(|(lhs, rhs)| lhs.wrapping_add(*rhs).wrapping_mul(*rhs))
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

    let bindings = wrapping_add_mul_i32_bindings();
    let builder = wrapping_add_mul_i32::<N>(&bindings)?;
    let kernel = builder.ir();
    let invocations = u32::try_from(N)?;
    let prepared = support::cold_once("PCU i32 cold prepare", || {
        backend.prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(invocations).expect("nonzero")),
        })
    })?;
    let source = format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_i32_add_mul(const unsigned int* left, const unsigned int* right, unsigned int* output) {{\n    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;\n    if (id < {N}u) output[id] = (left[id] + right[id]) * right[id];\n}}\n"
    );
    let image = compile_hip_source(&source, architecture)?;
    let module = runtime.load_module(&image)?;
    let native_function = module.function(c"native_i32_add_mul")?;
    let stream = runtime.create_stream()?;
    let native_arguments = [
        HipKernelArgument::Buffer(&native_left),
        HipKernelArgument::Buffer(&native_right),
        HipKernelArgument::Buffer(&native_output),
    ];
    let grid = invocations.div_ceil(BLOCK_SIZE);

    run_prepared_typed_binary(
        backend,
        &prepared,
        PcuValueType::i32(),
        &pcu_left,
        &pcu_right,
        &pcu_output,
    )?;
    run_direct(&native_function, &stream, &native_arguments, grid)?;
    let mut pcu_bytes = vec![0; left_bytes.len()];
    let mut native_bytes = vec![0; left_bytes.len()];
    pcu_output.copy_to(&mut pcu_bytes)?;
    native_output.copy_to(&mut native_bytes)?;
    let expected_bytes = encode(&expected);
    if pcu_bytes != expected_bytes || native_bytes != expected_bytes {
        return Err("i32 PCU/native wrapping Add/Mul output mismatch".into());
    }
    println!("i32 wrapping Add/Mul verified for {N} values, including overflow");

    {
        let mut group = criterion.benchmark_group("i32-wrapping-add-mul-chain");
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
                                PcuValueType::i32(),
                                &pcu_left,
                                &pcu_right,
                                &pcu_output,
                            )
                            .expect("PCU i32 launch failed");
                            let native =
                                run_direct(&native_function, &stream, &native_arguments, grid)
                                    .expect("native i32 launch failed");
                            (pcu, native)
                        } else {
                            let native =
                                run_direct(&native_function, &stream, &native_arguments, grid)
                                    .expect("native i32 launch failed");
                            let pcu = run_prepared_typed_binary(
                                backend,
                                &prepared,
                                PcuValueType::i32(),
                                &pcu_left,
                                &pcu_right,
                                &pcu_output,
                            )
                            .expect("PCU i32 launch failed");
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
    Ok(())
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = bench
}
criterion_main!(benches);

//! Exact binary32-to-binary64 widening through prepared PCU versus integer-only HIP.

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
    f32_bits_to_f64_bits,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCompletionOutcome,
    PcuDispatchControlOp,
    PcuDispatchConversion,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuInvocationShape,
    PcuKernelId,
    PcuOwnedBinding,
    PcuOwnedCompletion,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    compile_hip_source,
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmPreparedDispatch,
};
use support::selection;

const BLOCK: u32 = 64;
const INPUT: PcuBindingRef = PcuBindingRef::new(0, 0);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 1);
const SOURCE: PcuDispatchValueId = PcuDispatchValueId(1);
const WIDENED: PcuDispatchValueId = PcuDispatchValueId(2);

const HIP: &str = r#"
#include <hip/hip_runtime.h>
__device__ __forceinline__ unsigned long long exact_widen(unsigned int bits) {
  unsigned long long sign=(unsigned long long)(bits>>31u)<<63u;
  unsigned int exponent=(bits>>23u)&255u, fraction=bits&0x7fffffu;
  if(exponent==255u) return sign|(0x7ffull<<52u)|((unsigned long long)fraction<<29u);
  if(exponent) return sign|((unsigned long long)(exponent+896u)<<52u)|((unsigned long long)fraction<<29u);
  if(!fraction) return sign;
  unsigned int leading=0u; while((fraction>>(leading+1u))!=0u) ++leading;
  unsigned long long significand=(unsigned long long)fraction<<(52u-leading);
  return sign|((unsigned long long)(leading+874u)<<52u)|(significand&0x000fffffffffffffull);
}
extern "C" __global__ void native_exact(const unsigned int* in,unsigned long long* out,unsigned n) {
  unsigned id=blockIdx.x*blockDim.x+threadIdx.x; if(id<n) out[id]=exact_widen(in[id]);
}
extern "C" __global__ void native_exact_grid(const unsigned int* in,unsigned long long* out,unsigned n,unsigned stride) {
  unsigned id=blockIdx.x*blockDim.x+threadIdx.x; if(id<stride) for(unsigned long long i=id;i<n;i+=stride) out[i]=exact_widen(in[i]);
}
"#;

fn run(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect::<Vec<_>>();
    let (backend, selected) = selection::open_ranked(&discovery, candidates, BLOCK)?;
    let arch = selected
        .architecture
        .ok_or("selected device has no HIP architecture")?;
    let runtime = discovery.open_device(selected.device)?;
    println!(
        "f32→f64 exact device: {} ({arch})",
        discovery.device_info(selected.device)?.name
    );
    let module = runtime.load_module(&compile_hip_source(HIP, &arch)?)?;
    let direct = module.function(c"native_exact")?;
    let grid_kernel = module.function(c"native_exact_grid")?;
    for (extent, invocations, grid) in [
        (65, 65, false),
        (1 << 20, 1 << 20, false),
        (2048, 250, true),
    ] {
        run_case(
            c,
            &backend,
            &runtime,
            &direct,
            &grid_kernel,
            extent,
            invocations,
            grid,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)]
fn run_case(
    c: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    direct_kernel: &HipKernel,
    grid_kernel: &HipKernel,
    extent: u32,
    invocations: u32,
    grid: bool,
) -> Result<(), Box<dyn Error>> {
    let mut input = (0..extent)
        .map(|index| index.wrapping_mul(0x9e37_79b9).wrapping_add(0x7f00_0001))
        .collect::<Vec<_>>();
    for (index, edge) in [
        0,
        0x8000_0000,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0x7f80_1234,
        0xff80_0001,
        0xffc1_2345,
        0x7fc0_0000,
        0x3f80_0000,
        0xbf80_0000,
        u32::MAX,
    ]
    .into_iter()
    .enumerate()
    {
        if let Some(slot) = input.get_mut(index) {
            *slot = edge;
        }
    }
    let expected = input
        .iter()
        .map(|bits| f32_bits_to_f64_bits(*bits))
        .collect::<Vec<_>>();
    let input_bytes = input
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let expected_bytes = expected
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let mut pcu_input = backend.allocate(input_bytes.len())?;
    pcu_input.copy_from(&input_bytes)?;
    let pcu_output = backend.allocate(expected_bytes.len())?;
    let mut native_input = runtime.allocate(input_bytes.len())?;
    native_input.copy_from(&input_bytes)?;
    let native_output = runtime.allocate(expected_bytes.len())?;
    let bindings = [
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
            PcuValueType::f64(),
        ),
    ];
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: SOURCE,
            binding: INPUT,
            index: PcuDispatchIndex::GridStrideId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
            result: WIDENED,
            value: SOURCE,
            conversion: PcuDispatchConversion::F32ToF64Exact,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index: PcuDispatchIndex::GridStrideId,
            value: WIDENED,
        }),
    ];
    let direct_ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: SOURCE,
            binding: INPUT,
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
            result: WIDENED,
            value: SOURCE,
            conversion: PcuDispatchConversion::F32ToF64Exact,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index: PcuDispatchIndex::InvocationId,
            value: WIDENED,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "f32_f64_exact",
            logical_shape: [invocations, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &grid_ops } else { &direct_ops },
        type_caps: PcuValueTypeCaps::SCALAR_VALUES
            .union(PcuValueTypeCaps::FLOAT32)
            .union(PcuValueTypeCaps::FLOAT64),
        feature_caps: PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES),
    };
    let prepared = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(
            NonZeroU32::new(invocations).ok_or("zero invocations")?,
        ),
    })?;
    let refs = [
        backend.binding(
            INPUT,
            PcuBindingAccess::ReadOnly,
            fusion_pcu::PcuBindingType::Value(PcuValueType::f32()),
            pcu_input,
        )?,
        backend.binding(
            OUTPUT,
            PcuBindingAccess::WriteOnly,
            fusion_pcu::PcuBindingType::Value(PcuValueType::f64()),
            pcu_output.clone(),
        )?,
    ];
    let stream = runtime.create_stream()?;
    let n = extent.to_ne_bytes();
    let stride = invocations.to_ne_bytes();
    let args_direct = [
        HipKernelArgument::Buffer(&native_input),
        HipKernelArgument::Buffer(&native_output),
        HipKernelArgument::Bytes(&n),
    ];
    let args_grid = [
        HipKernelArgument::Buffer(&native_input),
        HipKernelArgument::Buffer(&native_output),
        HipKernelArgument::Bytes(&n),
        HipKernelArgument::Bytes(&stride),
    ];
    let args = if grid {
        &args_grid[..]
    } else {
        &args_direct[..]
    };
    let native = if grid { grid_kernel } else { direct_kernel };
    let blocks = invocations.div_ceil(BLOCK);
    let mut completion = prepared.submit(&refs)?;
    if completion.wait()? != PcuCompletionOutcome::Succeeded {
        return Err("PCU preflight failed".into());
    }
    dispatch_support::run_direct(native, &stream, args, blocks)?;
    let mut pcu_result = vec![0; expected_bytes.len()];
    let mut native_result = vec![0; expected_bytes.len()];
    pcu_output.copy_to(&mut pcu_result)?;
    native_output.copy_to(&mut native_result)?;
    if pcu_result != expected_bytes || native_result != expected_bytes {
        return Err("exact f32→f64 preflight mismatch".into());
    }
    let mut group = c.benchmark_group(format!(
        "f32-to-f64-exact-{}",
        if grid { "grid" } else { "direct" }
    ));
    group.throughput(Throughput::Elements(u64::from(extent)));
    group.bench_function(
        BenchmarkId::new("Prepared PCU", format!("n{extent}-i{invocations}")),
        |b| b.iter(|| black_box(run_pcu(&prepared, &refs))),
    );
    group.bench_function(
        BenchmarkId::new("Direct HIP", format!("n{extent}-i{invocations}")),
        |b| {
            b.iter(|| {
                black_box(
                    dispatch_support::run_direct(native, &stream, args, blocks)
                        .expect("HIP launch"),
                )
            });
        },
    );
    group.finish();
    Ok(())
}

fn run_pcu(prepared: &RocmPreparedDispatch, refs: &[PcuOwnedBinding<DeviceBuffer>]) -> Duration {
    let start = std::time::Instant::now();
    let mut completion = prepared.submit(refs).expect("PCU submit");
    assert_eq!(
        completion.wait().expect("PCU completion"),
        PcuCompletionOutcome::Succeeded
    );
    start.elapsed()
}

fn benchmarks(c: &mut Criterion) {
    run(c).expect("f32 to f64 benchmark failed");
}
criterion_group! { name = benches; config = support::criterion_config(); targets = benchmarks }
criterion_main!(benches);

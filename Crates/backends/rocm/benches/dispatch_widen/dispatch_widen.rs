//! Exact integer widening through prepared PCU versus independent HIP kernels.

extern crate pcu_facade as fusion_pcu;

#[path = "../support/dispatch.rs"]
#[allow(dead_code)]
mod dispatch_support;
#[allow(dead_code)]
#[path = "../support/support.rs"]
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
    HipStreamHandle,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmPreparedDispatch,
};
use support::selection;

const BLOCK_SIZE: u32 = 64;
const INPUT: PcuBindingRef = PcuBindingRef::new(0, 0);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 1);
const SOURCE: PcuDispatchValueId = PcuDispatchValueId(1);
const WIDENED: PcuDispatchValueId = PcuDispatchValueId(2);

#[derive(Clone, Copy)]
enum Width {
    I8I16,
    U8U16,
    I16I32,
    U16U32,
    I32I64,
    U32U64,
}

impl Width {
    const fn id(self) -> u32 {
        match self {
            Self::I8I16 => 1,
            Self::U8U16 => 2,
            Self::I16I32 => 3,
            Self::U16U32 => 4,
            Self::I32I64 => 5,
            Self::U32U64 => 6,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::I8I16 => "i8-to-i16",
            Self::U8U16 => "u8-to-u16",
            Self::I16I32 => "i16-to-i32",
            Self::U16U32 => "u16-to-u32",
            Self::I32I64 => "i32-to-i64",
            Self::U32U64 => "u32-to-u64",
        }
    }

    const fn types(
        self,
    ) -> (
        PcuValueType,
        PcuValueType,
        PcuDispatchConversion,
        PcuValueTypeCaps,
    ) {
        match self {
            Self::I8I16 => (
                PcuValueType::i8(),
                PcuValueType::i16(),
                PcuDispatchConversion::I8ToI16,
                PcuValueTypeCaps::SCALAR_VALUES
                    .union(PcuValueTypeCaps::INT8)
                    .union(PcuValueTypeCaps::INT16),
            ),
            Self::U8U16 => (
                PcuValueType::u8(),
                PcuValueType::u16(),
                PcuDispatchConversion::U8ToU16,
                PcuValueTypeCaps::SCALAR_VALUES
                    .union(PcuValueTypeCaps::UINT8)
                    .union(PcuValueTypeCaps::UINT16),
            ),
            Self::I16I32 => (
                PcuValueType::i16(),
                PcuValueType::i32(),
                PcuDispatchConversion::I16ToI32,
                PcuValueTypeCaps::SCALAR_VALUES
                    .union(PcuValueTypeCaps::INT16)
                    .union(PcuValueTypeCaps::INT32),
            ),
            Self::U16U32 => (
                PcuValueType::u16(),
                PcuValueType::u32(),
                PcuDispatchConversion::U16ToU32,
                PcuValueTypeCaps::SCALAR_VALUES
                    .union(PcuValueTypeCaps::UINT16)
                    .union(PcuValueTypeCaps::UINT32),
            ),
            Self::I32I64 => (
                PcuValueType::i32(),
                PcuValueType::i64(),
                PcuDispatchConversion::I32ToI64,
                PcuValueTypeCaps::SCALAR_VALUES
                    .union(PcuValueTypeCaps::INT32)
                    .union(PcuValueTypeCaps::INT64),
            ),
            Self::U32U64 => (
                PcuValueType::u32(),
                PcuValueType::u64(),
                PcuDispatchConversion::U32ToU64,
                PcuValueTypeCaps::SCALAR_VALUES
                    .union(PcuValueTypeCaps::UINT32)
                    .union(PcuValueTypeCaps::UINT64),
            ),
        }
    }

    const fn function(self, grid: bool) -> &'static std::ffi::CStr {
        match (self, grid) {
            (Self::I8I16, false) => c"widen_signed",
            (Self::I8I16, true) => c"widen_signed_grid",
            (Self::U8U16, false) => c"widen_unsigned",
            (Self::U8U16, true) => c"widen_unsigned_grid",
            (Self::I16I32, false) => c"widen_i16_i32",
            (Self::I16I32, true) => c"widen_i16_i32_grid",
            (Self::U16U32, false) => c"widen_u16_u32",
            (Self::U16U32, true) => c"widen_u16_u32_grid",
            (Self::I32I64, false) => c"widen_i32_i64",
            (Self::I32I64, true) => c"widen_i32_i64_grid",
            (Self::U32U64, false) => c"widen_u32_u64",
            (Self::U32U64, true) => c"widen_u32_u64_grid",
        }
    }
}

const HIP_SOURCE: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void widen_signed(const signed char* in, short* out, unsigned n) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) out[id] = static_cast<short>(in[id]);
}
extern "C" __global__ void widen_unsigned(const unsigned char* in, unsigned short* out, unsigned n) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) out[id] = static_cast<unsigned short>(in[id]);
}
extern "C" __global__ void widen_signed_grid(const signed char* in, short* out, unsigned n, unsigned stride) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < stride) for (; id < n; id += stride) out[id] = static_cast<short>(in[id]);
}
extern "C" __global__ void widen_unsigned_grid(const unsigned char* in, unsigned short* out, unsigned n, unsigned stride) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < stride) for (; id < n; id += stride) out[id] = static_cast<unsigned short>(in[id]);
}
extern "C" __global__ void widen_i16_i32(const short* in, int* out, unsigned n) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) out[id] = static_cast<int>(in[id]);
}
extern "C" __global__ void widen_u16_u32(const unsigned short* in, unsigned int* out, unsigned n) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) out[id] = static_cast<unsigned int>(in[id]);
}
extern "C" __global__ void widen_i32_i64(const int* in, long long* out, unsigned n) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) out[id] = static_cast<long long>(in[id]);
}
extern "C" __global__ void widen_u32_u64(const unsigned int* in, unsigned long long* out, unsigned n) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) out[id] = static_cast<unsigned long long>(in[id]);
}
extern "C" __global__ void widen_i16_i32_grid(const short* in, int* out, unsigned n, unsigned stride) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < stride) for (; id < n; id += stride) out[id] = static_cast<int>(in[id]);
}
extern "C" __global__ void widen_u16_u32_grid(const unsigned short* in, unsigned int* out, unsigned n, unsigned stride) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < stride) for (; id < n; id += stride) out[id] = static_cast<unsigned int>(in[id]);
}
extern "C" __global__ void widen_i32_i64_grid(const int* in, long long* out, unsigned n, unsigned stride) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < stride) for (; id < n; id += stride) out[id] = static_cast<long long>(in[id]);
}
extern "C" __global__ void widen_u32_u64_grid(const unsigned int* in, unsigned long long* out, unsigned n, unsigned stride) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < stride) for (; id < n; id += stride) out[id] = static_cast<unsigned long long>(in[id]);
}
"#;

fn inputs(extent: usize, width: Width) -> (Vec<u8>, Vec<u8>) {
    macro_rules! pack {
        ($ty:ty, $target:ty, [$($edge:expr),+ $(,)?]) => {{
            let edges: [$ty; 7] = [$($edge),+];
            let values = (0..extent).map(|index| edges[index % edges.len()]).collect::<Vec<_>>();
            let source = values.iter().flat_map(|value| value.to_ne_bytes()).collect();
            let expected = values.iter().map(|&value| <$target>::from(value))
                .flat_map(|value| value.to_ne_bytes()).collect();
            (source, expected)
        }};
    }
    match width {
        Width::I8I16 => pack!(i8, i16, [i8::MIN, -1, 0, 1, i8::MAX, -42, 42]),
        Width::U8U16 => pack!(u8, u16, [0, 1, 127, 128, u8::MAX, 42, 200]),
        Width::I16I32 => pack!(i16, i32, [i16::MIN, -1, 0, 1, i16::MAX, -12345, 23456]),
        Width::U16U32 => pack!(u16, u32, [0, 1, 0x7fff, 0x8000, u16::MAX, 12345, 54321]),
        Width::I32I64 => pack!(
            i32,
            i64,
            [i32::MIN, -1, 0, 1, i32::MAX, -1_234_567_890, 1_234_567_890]
        ),
        Width::U32U64 => pack!(
            u32,
            u64,
            [
                0,
                1,
                0x7fff_ffff,
                0x8000_0000,
                u32::MAX,
                1_234_567_890,
                3_000_000_000
            ]
        ),
    }
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
        "widen device: {} ({arch})",
        discovery.device_info(selected.device)?.name
    );
    let module = runtime.load_module(&compile_hip_source(HIP_SOURCE, &arch)?)?;
    for width in [
        Width::I8I16,
        Width::U8U16,
        Width::I16I32,
        Width::U16U32,
        Width::I32I64,
        Width::U32U64,
    ] {
        for extent in [65_u32, 1 << 20] {
            run_case(c, &backend, &runtime, &module, width, extent, extent, false)?;
        }
        run_case(c, &backend, &runtime, &module, width, 2048, 250, true)?;
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
    runtime: &HipRuntime,
    module: &fusion_pcu_rocm::HipModule,
    width: Width,
    extent: u32,
    invocations: u32,
    grid_stride: bool,
) -> Result<(), Box<dyn Error>> {
    let (source, expected) = inputs(usize::try_from(extent)?, width);
    let mut pcu_input = backend.allocate(source.len())?;
    let pcu_output = backend.allocate(expected.len())?;
    pcu_input.copy_from(&source)?;
    let mut native_input = runtime.allocate(source.len())?;
    let native_output = runtime.allocate(expected.len())?;
    native_input.copy_from(&source)?;
    let (input_ty, output_ty, conversion, type_caps) = width.types();
    let bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            input_ty,
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            output_ty,
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
            conversion,
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
            conversion,
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
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(width.id()),
        entry: PcuDispatchEntryPoint {
            name: "widen",
            logical_shape: [invocations, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid_stride { &grid_ops } else { &direct_ops },
        type_caps,
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
            fusion_pcu::PcuBindingType::Value(input_ty),
            pcu_input.clone(),
        )?,
        backend.binding(
            OUTPUT,
            PcuBindingAccess::WriteOnly,
            fusion_pcu::PcuBindingType::Value(output_ty),
            pcu_output.clone(),
        )?,
    ];
    let function = module.function(width.function(grid_stride))?;
    let stream = runtime.create_stream()?;
    let extent_bytes = extent.to_ne_bytes();
    let stride_bytes = invocations.to_ne_bytes();
    let direct_args = [
        HipKernelArgument::Buffer(&native_input),
        HipKernelArgument::Buffer(&native_output),
        HipKernelArgument::Bytes(&extent_bytes),
    ];
    let grid_args = [
        HipKernelArgument::Buffer(&native_input),
        HipKernelArgument::Buffer(&native_output),
        HipKernelArgument::Bytes(&extent_bytes),
        HipKernelArgument::Bytes(&stride_bytes),
    ];
    let args = if grid_stride {
        &grid_args[..]
    } else {
        &direct_args[..]
    };
    let grid = invocations.div_ceil(BLOCK_SIZE);
    let mut completion = prepared.submit(&refs)?;
    if completion.wait()? != PcuCompletionOutcome::Succeeded {
        return Err("PCU widening preflight failed".into());
    }
    dispatch_support::run_direct(&function, &stream, args, grid)?;
    let mut pcu_result = vec![0_u8; expected.len()];
    let mut native_result = vec![0_u8; expected.len()];
    pcu_output.copy_to(&mut pcu_result)?;
    native_output.copy_to(&mut native_result)?;
    if pcu_result != expected || native_result != expected {
        return Err(format!(
            "{} mismatch: extent={extent}, invocations={invocations}",
            width.label()
        )
        .into());
    }
    // Warm both routes before counting Rust heap calls on this benchmark thread.
    black_box(run_pcu(&prepared, &refs));
    black_box(run_native(&function, &stream, args, grid));
    let _capture = dispatch_support::AllocationCapture::start();
    black_box(run_pcu(&prepared, &refs));
    let pcu_allocations = dispatch_support::AllocationCapture::finish();
    let _capture = dispatch_support::AllocationCapture::start();
    black_box(run_native(&function, &stream, args, grid));
    let hip_allocations = dispatch_support::AllocationCapture::finish();
    println!(
        "{} verified: extent={extent}, invocations={invocations}; Rust heap PCU/HIP alloc+realloc {}/{}, bytes {}/{} (native runtime excluded)",
        width.label(),
        pcu_allocations.alloc_calls + pcu_allocations.realloc_calls,
        hip_allocations.alloc_calls + hip_allocations.realloc_calls,
        pcu_allocations.requested_bytes,
        hip_allocations.requested_bytes
    );
    let mut group = c.benchmark_group(format!(
        "{}-{}",
        width.label(),
        if grid_stride { "grid" } else { "direct" }
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

fn run_pcu(prepared: &RocmPreparedDispatch, refs: &[PcuOwnedBinding<DeviceBuffer>]) -> Duration {
    let start = std::time::Instant::now();
    let mut completion = prepared.submit(refs).expect("PCU submit");
    assert_eq!(
        completion.wait().expect("PCU completion"),
        PcuCompletionOutcome::Succeeded
    );
    start.elapsed()
}

fn run_native(
    function: &HipKernel,
    stream: &HipStreamHandle,
    args: &[HipKernelArgument<'_>],
    grid: u32,
) -> Duration {
    dispatch_support::run_direct(function, stream, args, grid)
        .expect("HIP launch")
        .total
}

fn benchmarks(c: &mut Criterion) {
    run(c).expect("widen benchmark failed");
}

criterion_group! { name = benches; config = support::criterion_config(); targets = benchmarks }
criterion_main!(benches);

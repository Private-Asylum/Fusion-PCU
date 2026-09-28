//! Numerical f32/f16/bf16 conversion through prepared PCU versus independent HIP kernels.

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
    PcuF16Bits,
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
const CONVERTED: PcuDispatchValueId = PcuDispatchValueId(2);

#[derive(Clone, Copy)]
enum Conversion {
    F32ToF16,
    F16ToF32,
    F32ToBf16,
    Bf16ToF32,
}

impl Conversion {
    const ALL: [Self; 4] = [
        Self::F32ToF16,
        Self::F16ToF32,
        Self::F32ToBf16,
        Self::Bf16ToF32,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::F32ToF16 => "f32-to-f16",
            Self::F16ToF32 => "f16-to-f32",
            Self::F32ToBf16 => "f32-to-bf16",
            Self::Bf16ToF32 => "bf16-to-f32",
        }
    }

    const fn types(self) -> (PcuValueType, PcuValueType, PcuDispatchConversion) {
        match self {
            Self::F32ToF16 => (
                PcuValueType::f32(),
                PcuValueType::f16(),
                PcuDispatchConversion::F32ToF16Bits,
            ),
            Self::F16ToF32 => (
                PcuValueType::f16(),
                PcuValueType::f32(),
                PcuDispatchConversion::F16BitsToF32,
            ),
            Self::F32ToBf16 => (
                PcuValueType::f32(),
                PcuValueType::bf16(),
                PcuDispatchConversion::F32ToBf16Bits,
            ),
            Self::Bf16ToF32 => (
                PcuValueType::bf16(),
                PcuValueType::f32(),
                PcuDispatchConversion::Bf16BitsToF32,
            ),
        }
    }

    const fn hip_name(self, grid: bool) -> &'static std::ffi::CStr {
        match (self, grid) {
            (Self::F32ToF16, false) => c"f32_to_f16",
            (Self::F16ToF32, false) => c"f16_to_f32",
            (Self::F32ToBf16, false) => c"f32_to_bf16",
            (Self::Bf16ToF32, false) => c"bf16_to_f32",
            (Self::F32ToF16, true) => c"f32_to_f16_grid",
            (Self::F16ToF32, true) => c"f16_to_f32_grid",
            (Self::F32ToBf16, true) => c"f32_to_bf16_grid",
            (Self::Bf16ToF32, true) => c"bf16_to_f32_grid",
        }
    }

    fn expected(self, input: &[u8]) -> Vec<u8> {
        match self {
            Self::F32ToF16 => input
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|chunk| {
                    let bits = u32::from_ne_bytes(*chunk);
                    PcuF16Bits::from_f32(f32::from_bits(bits))
                        .to_bits()
                        .to_ne_bytes()
                })
                .collect(),
            Self::F16ToF32 => input
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|chunk| {
                    let bits = u16::from_ne_bytes(*chunk);
                    PcuF16Bits::from_bits(bits).to_f32().to_bits().to_ne_bytes()
                })
                .collect(),
            Self::F32ToBf16 => input
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|chunk| {
                    let bits = u32::from_ne_bytes(*chunk);
                    PcuBf16Bits::from_f32(f32::from_bits(bits))
                        .to_bits()
                        .to_ne_bytes()
                })
                .collect(),
            Self::Bf16ToF32 => input
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|chunk| {
                    let bits = u16::from_ne_bytes(*chunk);
                    PcuBf16Bits::from_bits(bits)
                        .to_f32()
                        .to_bits()
                        .to_ne_bytes()
                })
                .collect(),
        }
    }
}

// Deliberately use explicit bit algorithms in the HIP peer: native hardware conversion
// intrinsics are free to choose different NaN payload behavior from PCU's documented contract.
const HIP_SOURCE: &str = r#"
#include <hip/hip_runtime.h>
__device__ __forceinline__ unsigned int rshift_rne(unsigned int v, unsigned int s) {
  if (s == 0u) return v; if (s >= 32u) return 0u;
  unsigned int t=v>>s, d=v&((1u<<s)-1u), h=1u<<(s-1u);
  return t+((d>h || (d==h && (t&1u)))?1u:0u);
}
__device__ __forceinline__ unsigned short f32_f16(float x) {
  unsigned int b=__builtin_bit_cast(unsigned int,x); unsigned short sign=(unsigned short)((b>>16u)&0x8000u);
  int e=(int)((b>>23u)&255u); unsigned int f=b&0x7fffffu;
  if(e==255){if(!f)return (unsigned short)(sign|0x7c00u);return (unsigned short)(sign|0x7c00u|(f>>13u)|0x0200u);}
  if(e==0)return sign; int u=e-127; unsigned int m=0x800000u|f;
  if(u>15)return (unsigned short)(sign|0x7c00u);
  if(u>=-14){unsigned int q=rshift_rne(m,13u); unsigned int he=q==0x800u?(unsigned int)(u+16):(unsigned int)(u+15); unsigned int hf=q==0x800u?0u:q&0x3ffu; if(he>=31u)return (unsigned short)(sign|0x7c00u);return (unsigned short)(sign|(he<<10u)|hf);}
  if(u < -25)return sign; return (unsigned short)(sign|rshift_rne(m,(unsigned int)(-u-1)));
}
__device__ __forceinline__ float f16_f32(unsigned short h) {
  unsigned int b=(unsigned int)h, sign=(b&0x8000u)<<16u, e=(b>>10u)&31u, f=b&1023u, out;
  if(e==0u){if(f==0u)out=sign;else{int ue=-14;while((f&0x400u)==0u){f<<=1u;--ue;}f&=0x3ffu;out=sign|((unsigned int)(ue+127)<<23u)|(f<<13u);}}
  else if(e==31u)out=sign|0x7f800000u|(f<<13u); else out=sign|((e+112u)<<23u)|(f<<13u);
  return __builtin_bit_cast(float,out);
}
__device__ __forceinline__ unsigned short f32_bf16(float x) {
  unsigned int b=__builtin_bit_cast(unsigned int,x), e=b&0x7f800000u, f=b&0x007fffffu;
  if(e==0x7f800000u && f){unsigned short u=(unsigned short)(b>>16u);return (unsigned short)((u&0x8000u)|(u&0x7fffu)|0x0040u);}
  unsigned int hi=b>>16u, lo=b&0xffffu; return (unsigned short)(hi+(lo>0x8000u || (lo==0x8000u && (hi&1u))));
}
__device__ __forceinline__ float bf16_f32(unsigned short h) { return __builtin_bit_cast(float,(unsigned int)h<<16u); }
#define KERNEL(NAME, IN, OUT, EXPR) \
extern "C" __global__ void NAME(const IN* in, OUT* out, unsigned n) { unsigned id=blockIdx.x*blockDim.x+threadIdx.x; if(id<n) out[id]=(EXPR); }
#define GRID(NAME, IN, OUT, EXPR) \
extern "C" __global__ void NAME(const IN* in, OUT* out, unsigned n, unsigned stride) { unsigned id=blockIdx.x*blockDim.x+threadIdx.x; if(id<stride) for(;id<n;id+=stride) out[id]=(EXPR); }
KERNEL(f32_to_f16,float,unsigned short,f32_f16(in[id]))
KERNEL(f16_to_f32,unsigned short,float,f16_f32(in[id]))
KERNEL(f32_to_bf16,float,unsigned short,f32_bf16(in[id]))
KERNEL(bf16_to_f32,unsigned short,float,bf16_f32(in[id]))
GRID(f32_to_f16_grid,float,unsigned short,f32_f16(in[id]))
GRID(f16_to_f32_grid,unsigned short,float,f16_f32(in[id]))
GRID(f32_to_bf16_grid,float,unsigned short,f32_bf16(in[id]))
GRID(bf16_to_f32_grid,unsigned short,float,bf16_f32(in[id]))
"#;

fn input_pattern(conversion: Conversion, count: usize) -> Vec<u8> {
    match conversion {
        Conversion::F16ToF32 | Conversion::Bf16ToF32 => (0..count)
            .flat_map(|index| {
                u16::try_from(index % 65_536)
                    .expect("u16 range")
                    .to_ne_bytes()
            })
            .collect(),
        Conversion::F32ToF16 | Conversion::F32ToBf16 => {
            const EDGE_BITS: [u32; 20] = [
                0,
                0x8000_0000,
                1,
                0x007f_ffff,
                0x0080_0000,
                0x3300_0000,
                0x3380_0000,
                0x387f_e000,
                0x3880_0000,
                0x3f80_1000,
                0x3f80_3000,
                0x3f81_8000,
                0x477f_e000,
                0x4780_0000,
                0x7f7f_ffff,
                0x7f80_0000,
                0xff80_0000,
                0x7f80_1234,
                0xff80_1234,
                0x7fc0_0001,
            ];
            (0..count)
                .flat_map(|index| {
                    let bits = if index < EDGE_BITS.len() {
                        EDGE_BITS[index]
                    } else {
                        // A deterministic sweep across every binary32 bit pattern, including
                        // exponent transitions and a broad sample of NaNs/infinities.
                        u32::try_from(index)
                            .expect("bounded")
                            .wrapping_mul(0x9e37_79b9)
                            .wrapping_add(0x7f00_0001)
                    };
                    bits.to_ne_bytes()
                })
                .collect()
        }
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
        "half numerical conversion device: {} ({arch})",
        discovery.device_info(selected.device)?.name
    );
    let module = runtime.load_module(&compile_hip_source(HIP_SOURCE, &arch)?)?;
    for conversion in Conversion::ALL {
        for extent in [65_u32, 1 << 20] {
            run_case(
                c, &backend, &runtime, &module, conversion, extent, extent, false,
            )?;
        }
        run_case(c, &backend, &runtime, &module, conversion, 2048, 250, true)?;
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
    conversion: Conversion,
    extent: u32,
    invocations: u32,
    grid_stride: bool,
) -> Result<(), Box<dyn Error>> {
    let input = input_pattern(conversion, usize::try_from(extent)?);
    let expected = conversion.expected(&input);
    let mut pcu_input = backend.allocate(input.len())?;
    let pcu_output = backend.allocate(expected.len())?;
    pcu_input.copy_from(&input)?;
    let mut native_input = runtime.allocate(input.len())?;
    let native_output = runtime.allocate(expected.len())?;
    native_input.copy_from(&input)?;
    let (input_ty, output_ty, operation) = conversion.types();
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
            result: CONVERTED,
            value: SOURCE,
            conversion: operation,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index: PcuDispatchIndex::GridStrideId,
            value: CONVERTED,
        }),
    ];
    let direct = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: SOURCE,
            binding: INPUT,
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
            result: CONVERTED,
            value: SOURCE,
            conversion: operation,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index: PcuDispatchIndex::InvocationId,
            value: CONVERTED,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid = [
        PcuDispatchOp::GridStrideLoop {
            extent,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        id: PcuKernelId(u32::try_from(
            Conversion::ALL
                .iter()
                .position(|item| item.label() == conversion.label())
                .expect("member"),
        )?),
        entry: PcuDispatchEntryPoint {
            name: conversion.label(),
            logical_shape: [invocations, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid_stride { &grid } else { &direct },
        type_caps: PcuValueTypeCaps::SCALAR_VALUES
            | PcuValueTypeCaps::FLOAT16
            | PcuValueTypeCaps::BFLOAT16,
        feature_caps: PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES,
    };
    let shape =
        PcuInvocationShape::invocations(NonZeroU32::new(invocations).ok_or("zero invocations")?);
    let prepared = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape,
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
    let function = module.function(conversion.hip_name(grid_stride))?;
    let stream = runtime.create_stream()?;
    let n_bytes = extent.to_ne_bytes();
    let stride_bytes = invocations.to_ne_bytes();
    let direct_args = [
        HipKernelArgument::Buffer(&native_input),
        HipKernelArgument::Buffer(&native_output),
        HipKernelArgument::Bytes(&n_bytes),
    ];
    let grid_args = [
        HipKernelArgument::Buffer(&native_input),
        HipKernelArgument::Buffer(&native_output),
        HipKernelArgument::Bytes(&n_bytes),
        HipKernelArgument::Bytes(&stride_bytes),
    ];
    let args = if grid_stride {
        &grid_args[..]
    } else {
        &direct_args[..]
    };
    let launch_grid = invocations.div_ceil(BLOCK_SIZE);

    let mut completion = prepared.submit(&refs)?;
    if completion.wait()? != PcuCompletionOutcome::Succeeded {
        return Err(format!("{} PCU preflight failed", conversion.label()).into());
    }
    dispatch_support::run_direct(&function, &stream, args, launch_grid)?;
    let mut pcu_result = vec![0; expected.len()];
    let mut native_result = vec![0; expected.len()];
    pcu_output.copy_to(&mut pcu_result)?;
    native_output.copy_to(&mut native_result)?;
    if pcu_result != expected || native_result != expected {
        return Err(format!(
            "{} mismatch: extent={extent}, invocations={invocations}",
            conversion.label()
        )
        .into());
    }
    println!(
        "{} verified: extent={extent}, invocations={invocations}; edge vectors included",
        conversion.label()
    );
    black_box(run_pcu(&prepared, &refs));
    black_box(run_native(&function, &stream, args, launch_grid));
    let _capture = dispatch_support::AllocationCapture::start();
    black_box(run_pcu(&prepared, &refs));
    let pcu_allocations = dispatch_support::AllocationCapture::finish();
    let _capture = dispatch_support::AllocationCapture::start();
    black_box(run_native(&function, &stream, args, launch_grid));
    let hip_allocations = dispatch_support::AllocationCapture::finish();
    println!(
        "{} Rust heap PCU/HIP alloc+realloc {}/{}, requested bytes {}/{} (benchmark-thread allocator; native runtime excluded)",
        conversion.label(),
        pcu_allocations.alloc_calls + pcu_allocations.realloc_calls,
        hip_allocations.alloc_calls + hip_allocations.realloc_calls,
        pcu_allocations.requested_bytes,
        hip_allocations.requested_bytes,
    );
    let mut group = c.benchmark_group(format!(
        "{}-{}",
        conversion.label(),
        if grid_stride { "grid" } else { "direct" }
    ));
    group.throughput(Throughput::Elements(u64::from(extent)));
    for (label, pcu) in [("Prepared PCU", true), ("Direct HIP", false)] {
        group.bench_function(
            BenchmarkId::new(label, format!("n{extent}-i{invocations}")),
            |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for iteration in 0..iterations {
                        let (pcu_time, native_time) = if iteration.is_multiple_of(2) {
                            (
                                run_pcu(&prepared, &refs),
                                run_native(&function, &stream, args, launch_grid),
                            )
                        } else {
                            let native = run_native(&function, &stream, args, launch_grid);
                            (run_pcu(&prepared, &refs), native)
                        };
                        elapsed += if pcu { pcu_time } else { native_time };
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
    prepared: &RocmPreparedDispatch,
    bindings: &[PcuOwnedBinding<DeviceBuffer>],
) -> Duration {
    let start = std::time::Instant::now();
    let mut completion = prepared.submit(bindings).expect("PCU submit");
    assert_eq!(
        completion.wait().expect("PCU wait"),
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
    run(c).expect("half conversion benchmark failed");
}
criterion_group! { name = benches; config = support::criterion_config(); targets = benchmarks }
criterion_main!(benches);

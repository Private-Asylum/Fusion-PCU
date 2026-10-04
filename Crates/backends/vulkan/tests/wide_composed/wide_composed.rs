//! Four-wide composition with independent arbitrary-precision checked step goldens.
extern crate pcu_facade as fusion_pcu;
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCheckedInteger,
    PcuCompoundArithmeticPolicy,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuScalar,
    PcuImplementationRequirements,
    PcuBindingAccess,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
trait Format: PcuCheckedInteger + std::fmt::Debug {
    const LABEL: &'static str;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn call<const N: usize>(
        grid: bool,
        input: &[Self],
        seed: &Self,
        stage: &mut [Self],
        output: &mut [Self],
    ) -> Result<(), PcuExecutionError>;
}
macro_rules! formats {
    ($($ty:ty => $label:literal, $module:ident;)+) => {$(
        impl Format for $ty {
            const LABEL: &'static str = $label;
            fn from_bytes(bytes: &[u8]) -> Self { Self::decode_le(bytes.try_into().unwrap()) }
            fn call<const N: usize>(grid: bool, input: &[Self], seed: &Self, stage: &mut [Self], output: &mut [Self]) -> Result<(), PcuExecutionError> {
                if grid { source::grid::<Self,N>(input, seed, stage, output) }
                else { source::direct::<Self,N>(input, seed, stage, output) }
            }
        }
    )+};
}
formats! {
    u8 => "u8", primitiveU8;
    i8 => "i8", primitiveI8;
    u16 => "u16", primitiveU16;
    i16 => "i16", primitiveI16;
    u32 => "u32", primitiveU32;
    i32 => "i32", primitiveI32;
    u64 => "u64", primitiveU64;
    i64 => "i64", primitiveI64;
    u128 => "u128", primitiveU128;
    i128 => "i128", primitiveI128;
    fusion_pcu::PcuU256 => "u256", unsigned256;
    fusion_pcu::PcuI256 => "i256", signed256;
    fusion_pcu::PcuU512 => "u512", unsigned512;
    fusion_pcu::PcuI512 => "i512", signed512;
}
fn decode<T: Format>(hex: &str) -> T {
    let bytes = hex
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    T::from_bytes(&bytes)
}
fn small<T: Format>(value: u8) -> T {
    let mut bytes = vec![0; T::ENCODED_SIZE];
    bytes[0] = value;
    T::from_bytes(&bytes)
}
fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn verify<T: Format>(range: PcuRangePolicy) {
    const N: usize = 65;
    for line in include_str!("vectors.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields[0] != T::LABEL
            || fields[1]
                != if range == PcuRangePolicy::Clamp {
                    "1"
                } else {
                    "0"
                }
        {
            continue;
        }
        let input = decode::<T>(fields[2]);
        let seed = decode::<T>(fields[3]);
        let stage_want = decode::<T>(fields[4]);
        let output_want = decode::<T>(fields[5]);
        let code = fields[6].parse::<u32>().unwrap();
        for grid in [false, true] {
            let mut inputs = [small::<T>(0); N];
            inputs[7] = input;
            let sentinel = small::<T>(17);
            let mut stage = [sentinel; N + 2];
            let mut output = [sentinel; N + 3];
            let result = T::call::<N>(grid, &inputs, &seed, &mut stage, &mut output);
            if code == 0 {
                result.unwrap();
            } else {
                let fault = result.unwrap_err().arithmetic_fault().unwrap();
                assert_eq!(
                    (fault.invocation_id, fault.kind, fault.recovered),
                    (
                        7,
                        if code == 3 {
                            PcuExecutionFaultKind::ArithmeticOverflow
                        } else {
                            PcuExecutionFaultKind::ArithmeticUnderflow
                        },
                        range == PcuRangePolicy::Clamp
                    )
                );
            }
            if code != 0 && range == PcuRangePolicy::Reject {
                bits(&stage, &[sentinel; N + 2]);
                bits(&output, &[sentinel; N + 3]);
            } else {
                bits(&stage[7..8], &[stage_want]);
                bits(&output[7..8], &[output_want]);
                bits(&stage[N..], &[sentinel; 2]);
                bits(&output[N..], &[sentinel; 3]);
            }
            T::call::<N>(
                grid,
                &[small::<T>(0); N],
                &small::<T>(1),
                &mut stage,
                &mut output,
            )
            .unwrap();
            bits(&stage[..N], &[small::<T>(1); N]);
            bits(&output[..N], &[small::<T>(0); N]);
            let before_stage = stage;
            let before_output = output;
            assert!(T::call::<N>(grid, &inputs[..N - 1], &seed, &mut stage, &mut output).is_err());
            bits(&stage, &before_stage);
            bits(&output, &before_output);
        }
    }
}
fn verify_backend(backend: global::PcuBackendChoice) {
    if backend == global::PcuBackendChoice::Vulkan
        && let Some(directory) = std::env::var_os("PCU_COMPOSED_PACKAGES")
    {
        global::vulkan::configure(global::vulkan::PcuVulkanShaderOptions {
            source: fusion_pcu_vulkan::PcuVulkanShaderSource::ExternalComposed {
                directory: directory.into(),
                retain_in_memory: true,
            },
            cache: fusion_pcu_vulkan::PcuVulkanShaderCachePolicy::MemoryOnly,
        })
        .unwrap();
    }
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        global::configure(global::PcuExecutionPolicy {
                            backend,
                            numerical_mode,
                            float_underflow,
                            range_policy,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                ..Default::default()
                            },
                            ..Default::default()
                        })
                        .unwrap();
                        verify::<fusion_pcu::PcuU256>(range_policy);
                        verify::<fusion_pcu::PcuI256>(range_policy);
                        verify::<fusion_pcu::PcuU512>(range_policy);
                        verify::<fusion_pcu::PcuI512>(range_policy);
                        global::clear_thread_cache().unwrap();
                    }
                }
            }
        }
    }
}
fn saturation<T: Format>(kind: PcuExecutionFaultKind) -> T {
    let signed = T::LABEL.starts_with('i');
    let mut bytes = vec![0; T::ENCODED_SIZE];
    if kind == PcuExecutionFaultKind::ArithmeticOverflow {
        bytes.fill(255);
        if signed {
            *bytes.last_mut().unwrap() = 127;
        }
    } else if signed {
        *bytes.last_mut().unwrap() = 128;
    }
    T::from_bytes(&bytes)
}
fn reference<T: Format>() {
    for line in include_str!("vectors.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields[0] != T::LABEL {
            continue;
        }
        let a = decode::<T>(fields[2]);
        let b = decode::<T>(fields[3]);
        let clamp = fields[1] == "1";
        let mut first_fault = None;
        let mut apply = |result: Result<T, PcuExecutionFaultKind>| match result {
            Ok(value) => value,
            Err(kind) => {
                first_fault.get_or_insert(kind);
                saturation::<T>(kind)
            }
        };
        let stage = apply(a.pcu_checked_add(b));
        let product = apply(stage.pcu_checked_mul(a));
        let output = apply(product.pcu_checked_sub(a));
        let code = fields[6].parse::<u32>().unwrap();
        assert_eq!(
            first_fault,
            match code {
                0 => None,
                3 => Some(PcuExecutionFaultKind::ArithmeticOverflow),
                4 => Some(PcuExecutionFaultKind::ArithmeticUnderflow),
                _ => unreachable!(),
            }
        );
        if clamp || code == 0 {
            bits(&[stage], &[decode::<T>(fields[4])]);
            bits(&[output], &[decode::<T>(fields[5])]);
        }
    }
}
#[test]
fn independent_fixed_vectors_match_primitive_checked_law() {
    reference::<fusion_pcu::PcuU256>();
    reference::<fusion_pcu::PcuI256>();
    reference::<fusion_pcu::PcuU512>();
    reference::<fusion_pcu::PcuI512>();
}
#[test]
fn cpu_four_wide_composed_source_matches_independent_goldens() {
    // Added after the frozen CUDA native cut: CPU now admits this bounded profile.
    verify_backend(global::PcuBackendChoice::Cpu);
}
#[test]
#[ignore = "actual Vulkan device required: four-wide composition, ordered store/reload, independent goldens"]
fn actual_four_wide_composed_source() {
    verify_backend(global::PcuBackendChoice::Vulkan);
}

fn write_only<T: Format>(backend: &fusion_pcu_vulkan::PcuVulkanBackend, grid: bool) {
    const N: usize = 65;
    let mut bindings = source::direct_bindings::<T>();
    bindings[3].access = PcuBindingAccess::WriteOnly;
    let request = PcuImplementationRequirements::default();
    let check = |ir: &fusion_pcu::PcuDispatchKernelIr<'_>| {
        let profile = fusion_pcu_spirv::validate_composed_integer_map(ir).unwrap();
        assert!(profile.is_integer());
        assert_eq!(profile.element_bytes(), T::ENCODED_SIZE);
        assert_eq!(
            profile.declaration_access(3),
            Some(PcuBindingAccess::WriteOnly)
        );
        let schema = fusion_pcu::assess_checked_integer_map_resources::<4>(
            ir,
            ir.bindings[0].value_type().unwrap(),
            fusion_pcu::PcuValueTypeCaps::for_scalar(profile.scalar()),
        )
        .unwrap();
        assert_eq!(schema.logical_extent, 65);
        let output = schema.resource(PcuBindingRef::new(0, 3)).unwrap();
        assert_eq!(output.minimum_read_elements, 0);
        assert_eq!(output.minimum_write_elements, 65);
        let mut prepared = backend.prepare_host_kernel(ir).unwrap();
        let input = [small::<T>(1); N];
        let seed = [small::<T>(1)];
        let mut stage = [small::<T>(17); N + 2];
        let mut output = [small::<T>(17); N + 3];
        let mut call = |input: &[T], stage: &mut [T], output: &mut [T]| {
            prepared.call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &seed),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), stage),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 3), output),
            ])
        };
        call(&input, &mut stage, &mut output).unwrap();
        bits(&stage[..N], &[small::<T>(2); N]);
        bits(&output[..N], &[small::<T>(1); N]);
        bits(&stage[N..], &[small::<T>(17); 2]);
        bits(&output[N..], &[small::<T>(17); 3]);
        let before = (stage, output);
        assert!(call(&input[..N - 1], &mut stage, &mut output).is_err());
        bits(&stage, &before.0);
        bits(&output, &before.1);
        call(&input, &mut stage, &mut output).unwrap();
    };
    if grid {
        source::__grid_ir_with_float_underflow_policy::<T, N>(
            &bindings,
            request.float_underflow,
            request.range_policy,
            request,
        )
        .unwrap()
        .with_ir(check);
    } else {
        source::__direct_ir_with_float_underflow_policy::<T, N>(
            &bindings,
            request.float_underflow,
            request.range_policy,
            request,
        )
        .unwrap()
        .with_ir(check);
    }
}
#[test]
#[ignore = "actual Vulkan device required: write-only fourteen-width output and transaction preflight"]
fn actual_write_only_outputs() {
    let backend = fusion_pcu_vulkan::PcuVulkanBackend::new().unwrap();
    if let Some(directory) = std::env::var_os("PCU_COMPOSED_PACKAGES") {
        backend.configure_shader_source(
            fusion_pcu_vulkan::PcuVulkanShaderSource::ExternalComposed {
                directory: directory.into(),
                retain_in_memory: true,
            },
        );
    }
    for grid in [false, true] {
        write_only::<u8>(&backend, grid);
        write_only::<i8>(&backend, grid);
        write_only::<u16>(&backend, grid);
        write_only::<i16>(&backend, grid);
        write_only::<u32>(&backend, grid);
        write_only::<i32>(&backend, grid);
        write_only::<u64>(&backend, grid);
        write_only::<i64>(&backend, grid);
        write_only::<u128>(&backend, grid);
        write_only::<i128>(&backend, grid);
        write_only::<fusion_pcu::PcuU256>(&backend, grid);
        write_only::<fusion_pcu::PcuI256>(&backend, grid);
        write_only::<fusion_pcu::PcuU512>(&backend, grid);
        write_only::<fusion_pcu::PcuI512>(&backend, grid);
    }
}

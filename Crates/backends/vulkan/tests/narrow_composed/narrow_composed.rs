//! Normal ten-width staged composition against independent fixed goldens.
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
    PcuReproducibility,
    PcuScalar,
    PcuBindingRef,
    PcuImplementationRequirements,
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
    u8 => "u8", unsigned8;
    i8 => "i8", signed8;
    u16 => "u16", unsigned16;
    i16 => "i16", signed16;
    u32 => "u32", unsigned32;
    i32 => "i32", signed32;
    u64 => "u64", unsigned64;
    i64 => "i64", signed64;
    u128 => "u128", unsigned128;
    i128 => "i128", signed128;
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
fn check_source_request<T: Format>(request: PcuImplementationRequirements) {
    let bindings = source::direct_bindings::<T>();
    let check = |ir: &fusion_pcu::PcuDispatchKernelIr<'_>| {
        let profile = fusion_pcu_spirv::validate_composed_integer_map(ir).unwrap();
        assert_eq!(profile.requirements(), request);
        assert_eq!(profile.element_bytes(), T::ENCODED_SIZE);
        assert_eq!(
            profile.packing_lanes(),
            if T::ENCODED_SIZE < 4 {
                u32::try_from(4 / T::ENCODED_SIZE).unwrap()
            } else {
                1
            }
        );
        assert_eq!(ir.numerical_requirements, request);
        let schema = fusion_pcu::assess_checked_integer_map_resources::<4>(
            ir,
            ir.bindings[0].value_type().unwrap(),
            fusion_pcu::PcuValueTypeCaps::for_scalar(T::TYPE),
        )
        .unwrap();
        assert_eq!(schema.requirements, request);
        assert_eq!(schema.logical_extent, 65);
        assert_eq!(schema.resources().len(), 4);
        assert_eq!(
            schema
                .resource(PcuBindingRef::new(0, 1))
                .unwrap()
                .minimum_read_elements,
            1
        );
        let stage = schema.resource(PcuBindingRef::new(0, 2)).unwrap();
        assert_eq!(
            (
                stage.minimum_read_elements,
                stage.minimum_initial_read_elements,
                stage.minimum_write_elements
            ),
            (65, 0, 65)
        );
    };
    source::__direct_ir_with_float_underflow_policy::<T, 65>(
        &bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap()
    .with_ir(check);
    source::__grid_ir_with_float_underflow_policy::<T, 65>(
        &bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap()
    .with_ir(check);
    check_dead_source_request::<T>(request);
}
fn check_dead_source_request<T: Format>(request: PcuImplementationRequirements) {
    let dead_bindings = source::dead_direct_bindings::<T>();
    let check_dead = |ir: &fusion_pcu::PcuDispatchKernelIr<'_>| {
        let profile = fusion_pcu_spirv::validate_one_effect_integer_map(ir).unwrap();
        assert_eq!(profile.requirements(), request);
        assert!(profile.is_one_effect());
        assert_eq!(ir.numerical_requirements, request);
        let schema = fusion_pcu::assess_checked_integer_map_resources::<4>(
            ir,
            ir.bindings[0].value_type().unwrap(),
            fusion_pcu::PcuValueTypeCaps::for_scalar(T::TYPE),
        )
        .unwrap();
        assert_eq!(schema.resources().len(), 3);
        assert_eq!(schema.logical_extent, 65);
        let body = match ir.ops.first() {
            Some(fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. }) => *body,
            _ => ir.ops,
        };
        assert_eq!(
            body.iter()
                .filter(|op| matches!(
                    op,
                    fusion_pcu::PcuDispatchOp::Data(
                        fusion_pcu::PcuDispatchDataOp::CheckedIntegerBinary { .. }
                    )
                ))
                .count(),
            1
        );
    };
    source::__dead_direct_ir_with_float_underflow_policy::<T, 65>(
        &dead_bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap()
    .with_ir(check_dead);
    source::__dead_grid_ir_with_float_underflow_policy::<T, 65>(
        &dead_bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap()
    .with_ir(check_dead);
}
fn verify<T: Format>(range: PcuRangePolicy, request: PcuImplementationRequirements) {
    const N: usize = 65;
    assert_eq!(request.range_policy, range);
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
                let mut expected_stage = [seed; N];
                let mut expected_output = [small::<T>(0); N];
                expected_stage[7] = stage_want;
                expected_output[7] = output_want;
                bits(&stage[..N], &expected_stage);
                bits(&output[..N], &expected_output);
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
    verify_dead::<T>(range);
}
fn verify_dead<T: Format>(range: PcuRangePolicy) {
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
        let fault_kind = input.pcu_checked_add(seed).err();
        let mut inputs = [small::<T>(0); N];
        inputs[7] = input;
        let sentinel = small::<T>(17);
        for grid in [false, true] {
            let mut output = [sentinel; N + 3];
            let call = |input: &[T], seed: &T, output: &mut [T]| {
                if grid {
                    source::dead_grid::<T, N>(input, seed, output)
                } else {
                    source::dead_direct::<T, N>(input, seed, output)
                }
            };
            let result = call(&inputs, &seed, &mut output);
            if let Some(kind) = fault_kind {
                let fault = result.unwrap_err().arithmetic_fault().unwrap();
                assert_eq!(
                    (fault.invocation_id, fault.kind, fault.recovered),
                    (7, kind, range == PcuRangePolicy::Clamp)
                );
            } else {
                result.unwrap();
            }
            if fault_kind.is_some() && range == PcuRangePolicy::Reject {
                bits(&output, &[sentinel; N + 3]);
            } else {
                bits(&output[..N], &inputs);
                bits(&output[N..], &[sentinel; 3]);
            }
            call(&[small::<T>(0); N], &small::<T>(1), &mut output).unwrap();
            bits(&output[..N], &[small::<T>(0); N]);
            let before = output;
            assert!(call(&inputs[..N - 1], &seed, &mut output).is_err());
            bits(&output, &before);
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
                                reproducibility: PcuReproducibility::Unspecified,
                            },
                            ..Default::default()
                        })
                        .unwrap();
                        let request = PcuImplementationRequirements {
                            numerical_mode,
                            float_underflow,
                            range_policy,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                reproducibility: PcuReproducibility::Unspecified,
                            },
                        };
                        macro_rules! widths {($($ty:ty),+)=>{$(check_source_request::<$ty>(request);verify::<$ty>(range_policy,request);)+};}
                        widths!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);
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
    reference::<u8>();
    reference::<i8>();
    reference::<u16>();
    reference::<i16>();
    reference::<u32>();
    reference::<i32>();
    reference::<u64>();
    reference::<i64>();
    reference::<u128>();
    reference::<i128>();
}
#[test]
fn cpu_normal_ten_width_composed_source_matches_independent_goldens() {
    // Independent normal ten-width source reference; prior wide source cuts stay frozen.
    verify_backend(global::PcuBackendChoice::Cpu);
}
#[test]
#[ignore = "actual Vulkan device required: ten-width composition, ordered store/reload, independent goldens"]
fn actual_normal_ten_width_composed_source() {
    verify_backend(global::PcuBackendChoice::Vulkan);
}

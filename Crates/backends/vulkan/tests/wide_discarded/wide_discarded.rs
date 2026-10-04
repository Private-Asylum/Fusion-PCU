//! Separately qualified four-wide discarded Add; wider one-effect operations remain refused.
mod source;
#[rustfmt::skip]
use pcu_facade::{global,PcuBindingRef,PcuCheckedInteger,PcuCompoundArithmeticPolicy,PcuDispatchDataOp,PcuDispatchKernelIr,PcuDispatchOp,PcuExecutionError,PcuExecutionFaultKind,PcuFloatUnderflowPolicy,PcuImplementationRequirements,PcuNumericalMode,PcuNumericalOptions,PcuPrecisionPolicy,PcuRangePolicy,PcuReproducibility,PcuScalar};
trait Format: PcuCheckedInteger {
    const LABEL: &'static str;
    fn from_bytes(bytes: &[u8]) -> Self;
}
macro_rules! formats {
    ($($ty:ty => $label:literal;)+) => {$(impl Format for $ty {
        const LABEL: &'static str = $label;
        fn from_bytes(bytes: &[u8]) -> Self { Self::decode_le(bytes.try_into().unwrap()) }
    })+};
}
formats! {
    pcu_facade::PcuU256 => "u256";
    pcu_facade::PcuI256 => "i256";
    pcu_facade::PcuU512 => "u512";
    pcu_facade::PcuI512 => "i512";
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
fn call<T: Format>(
    portable: bool,
    grid: bool,
    input: &[T],
    seed: &T,
    output: &mut [T],
) -> Result<(), PcuExecutionError> {
    match (portable, grid) {
        (false, false) => source::normal_direct::<T, 65>(input, seed, output),
        (false, true) => source::normal_grid::<T, 65>(input, seed, output),
        (true, false) => source::portable_direct::<T, 65>(input, seed, output),
        (true, true) => source::portable_grid::<T, 65>(input, seed, output),
    }
}
fn check_source<T: Format>(request: PcuImplementationRequirements) {
    let bindings = source::normal_direct_bindings::<T>();
    let check = |ir: &PcuDispatchKernelIr<'_>| {
        assert_eq!(ir.numerical_requirements, request);
        let body = ir
            .ops
            .iter()
            .find_map(|op| {
                if let PcuDispatchOp::GridStrideLoop { body, .. } = op {
                    Some(*body)
                } else {
                    None
                }
            })
            .unwrap_or(ir.ops);
        assert_eq!(
            body.iter()
                .filter(|op| matches!(
                    op,
                    PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { .. })
                ))
                .count(),
            1
        );
        if request.numerical_options.reproducibility == PcuReproducibility::PortableV1 {
            let schema =
                pcu_facade::describe_portable_v1_checked_integer_composed_map::<4>(ir).unwrap();
            assert_eq!(schema.requirements, request);
            assert_eq!(schema.logical_extent, 65);
            assert_eq!(schema.resources().len(), 3);
            assert_eq!(
                schema
                    .resource(PcuBindingRef::new(0, 1))
                    .unwrap()
                    .minimum_read_elements,
                1
            );
            assert_eq!(
                schema
                    .resource(PcuBindingRef::new(0, 2))
                    .unwrap()
                    .minimum_initial_read_elements,
                0
            );
        }
    };
    if request.numerical_options.reproducibility == PcuReproducibility::PortableV1 {
        source::__portable_direct_ir_with_float_underflow_policy::<T, 65>(
            &bindings,
            request.float_underflow,
            request.range_policy,
            request,
        )
        .unwrap()
        .with_ir(check);
        source::__portable_grid_ir_with_float_underflow_policy::<T, 65>(
            &bindings,
            request.float_underflow,
            request.range_policy,
            request,
        )
        .unwrap()
        .with_ir(check);
    } else {
        source::__normal_direct_ir_with_float_underflow_policy::<T, 65>(
            &bindings,
            request.float_underflow,
            request.range_policy,
            request,
        )
        .unwrap()
        .with_ir(check);
        source::__normal_grid_ir_with_float_underflow_policy::<T, 65>(
            &bindings,
            request.float_underflow,
            request.range_policy,
            request,
        )
        .unwrap()
        .with_ir(check);
    }
}
fn verify<T: Format>(request: PcuImplementationRequirements) {
    check_source::<T>(request);
    for row in include_str!("vectors.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let f = row.split_whitespace().collect::<Vec<_>>();
        if f[0] != T::LABEL || (f[1] == "1") != (request.range_policy == PcuRangePolicy::Clamp) {
            continue;
        }
        let input = decode::<T>(f[2]);
        let seed = decode::<T>(f[3]);
        let code = f[4].parse::<u32>().unwrap();
        let kind = match code {
            0 => None,
            3 => Some(PcuExecutionFaultKind::ArithmeticOverflow),
            4 => Some(PcuExecutionFaultKind::ArithmeticUnderflow),
            _ => unreachable!(),
        };
        assert_eq!(
            input.pcu_checked_add(seed).err(),
            kind,
            "independent exact Add golden"
        );
        for grid in [false, true] {
            let mut inputs = [small::<T>(0); 65];
            inputs[7] = input;
            let sentinel = small::<T>(17);
            let mut output = [sentinel; 68];
            let portable =
                request.numerical_options.reproducibility == PcuReproducibility::PortableV1;
            let result = call(portable, grid, &inputs, &seed, &mut output);
            if let Some(kind) = kind {
                let fault = result.unwrap_err().arithmetic_fault().unwrap();
                assert_eq!(
                    (fault.invocation_id, fault.kind, fault.recovered),
                    (7, kind, request.range_policy == PcuRangePolicy::Clamp)
                );
            } else {
                result.unwrap();
            }
            if kind.is_some() && request.range_policy == PcuRangePolicy::Reject {
                bits(&output, &[sentinel; 68]);
            } else {
                bits(&output[..65], &inputs);
                bits(&output[65..], &[sentinel; 3]);
            }
            call(
                portable,
                grid,
                &[small::<T>(0); 65],
                &small::<T>(1),
                &mut output,
            )
            .unwrap();
            bits(&output[..65], &[small::<T>(0); 65]);
            let before = output;
            assert!(call(portable, grid, &inputs[..64], &seed, &mut output).is_err());
            bits(&output, &before);
        }
    }
}
fn matrix(backend: global::PcuBackendChoice) {
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
                for reproducibility in [
                    PcuReproducibility::Unspecified,
                    PcuReproducibility::PortableV1,
                ] {
                    for float_underflow in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    ] {
                        for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                            let numerical_options = PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                reproducibility,
                            };
                            global::configure(global::PcuExecutionPolicy {
                                backend,
                                numerical_mode,
                                float_underflow,
                                range_policy,
                                numerical_options,
                                ..Default::default()
                            })
                            .unwrap();
                            let request = PcuImplementationRequirements {
                                numerical_mode,
                                float_underflow,
                                range_policy,
                                numerical_options,
                            };
                            verify::<pcu_facade::PcuU256>(request);
                            verify::<pcu_facade::PcuI256>(request);
                            verify::<pcu_facade::PcuU512>(request);
                            verify::<pcu_facade::PcuI512>(request);
                            global::clear_thread_cache().unwrap();
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn independent_wide_add_goldens_and_cpu_source_reference() {
    matrix(global::PcuBackendChoice::Cpu);
}
#[test]
#[ignore = "requires actual Vulkan device; qualified wide discarded Add source"]
fn actual_wide_discarded_source() {
    matrix(global::PcuBackendChoice::Vulkan);
}

#[test]
fn wide_one_effect_scope_accepts_add_and_refuses_sub_mul() {
    fn check<T: Format>() {
        let bindings = source::normal_direct_bindings::<T>();
        let check_ir = |ir: &PcuDispatchKernelIr<'_>| {
            assert!(fusion_pcu_spirv::validate_one_effect_integer_map(ir).is_ok());
            for operation in [
                pcu_facade::PcuDispatchIntegerBinaryOp::Sub,
                pcu_facade::PcuDispatchIntegerBinaryOp::Mul,
            ] {
                let mut ops = ir.ops.to_vec();
                for op in &mut ops {
                    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                        op, ..
                    }) = op
                    {
                        *op = operation;
                    }
                }
                let mut changed = *ir;
                changed.ops = &ops;
                if changed
                    .numerical_requirements
                    .numerical_options
                    .reproducibility
                    == PcuReproducibility::PortableV1
                {
                    assert!(
                        pcu_facade::describe_portable_v1_checked_integer_composed_map::<4>(
                            &changed
                        )
                        .is_ok()
                    );
                }
                assert!(fusion_pcu_spirv::validate_one_effect_integer_map(&changed).is_err());
            }
        };
        source::normal_direct_ir::<T, 65>(&bindings)
            .unwrap()
            .with_ir(check_ir);
        source::portable_direct_ir::<T, 65>(&bindings)
            .unwrap()
            .with_ir(check_ir);
    }
    check::<pcu_facade::PcuU256>();
    check::<pcu_facade::PcuI256>();
    check::<pcu_facade::PcuU512>();
    check::<pcu_facade::PcuI512>();
}

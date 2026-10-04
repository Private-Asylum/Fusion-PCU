//! Genuine source, typed graph and independent oracles for ordinary composed preparation.
extern crate pcu_facade as fusion_pcu;
#[path = "../../prepared/composed/tests/graph/graph.rs"]
mod graph;
#[path = "oracle/oracle.rs"]
#[allow(dead_code)]
// Retain the independently qualified low-format model and bounded native controls.
mod oracle;
#[path = "source/source.rs"]
#[allow(dead_code)] // Standalone ordinary companions are exercised by the facade source gate.
mod source;
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanPreparedHost,
    PcuVulkanError,
    PcuVulkanPreparedComposed,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuCompoundArithmeticPolicy,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy as Uf,
    PcuHostArgument,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuPreparedHostKernel,
    PcuRangePolicy as Range,
};
use oracle::Format;
#[path = "offers/offers.rs"]
mod offers;
#[path = "one_effect/one_effect.rs"]
mod one_effect;
#[path = "operations/operations.rs"]
mod operations;
#[path = "shadow/shadow.rs"]
mod shadow;

const UF: [Uf; 3] = [
    Uf::IeeeAfterRounding,
    Uf::RejectSubnormalResult,
    Uf::AllowGradualUnderflow,
];
type Entry<'a, T> = Box<dyn FnMut(&mut [T], &[T]) -> Result<(), PcuVulkanError> + 'a>;
fn source_plan<T: Format, const N: usize>(
    backend: &PcuVulkanBackend,
    uf: Uf,
    range: Range,
) -> Entry<'_, T> {
    match (uf, range) {
        (Uf::IeeeAfterRounding, Range::Reject) => {
            Box::new(source::reject_ieee_prepare::<T, N, _>(backend).unwrap())
        }
        (Uf::AllowGradualUnderflow, Range::Reject) => {
            Box::new(source::reject_gradual_prepare::<T, N, _>(backend).unwrap())
        }
        (Uf::RejectSubnormalResult, Range::Reject) => {
            Box::new(source::reject_tight_prepare::<T, N, _>(backend).unwrap())
        }
        (Uf::IeeeAfterRounding, Range::Clamp) => {
            Box::new(source::clamp_ieee_prepare::<T, N, _>(backend).unwrap())
        }
        (Uf::AllowGradualUnderflow, Range::Clamp) => {
            Box::new(source::clamp_gradual_prepare::<T, N, _>(backend).unwrap())
        }
        (Uf::RejectSubnormalResult, Range::Clamp) => {
            Box::new(source::clamp_tight_prepare::<T, N, _>(backend).unwrap())
        }
    }
}
fn observed(result: Result<(), PcuVulkanError>) -> Result<(), PcuExecutionFault> {
    result.map_err(|error| match error {
        PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected {other:?}"),
    })
}
fn graph_plan<T: Format, const N: usize>(
    backend: &PcuVulkanBackend,
    uf: Uf,
    range: Range,
) -> PcuVulkanPreparedComposed {
    match graph::prepare::<T, N, _>(backend, uf, range) {
        PcuVulkanPreparedHost::Composed(plan) => *plan,
        _ => panic!("typed composed graph selected a different prepared family"),
    }
}
fn explicit<T: Format>(
    plan: &mut PcuVulkanPreparedComposed,
    input: &[T],
    output: &mut [T],
) -> Result<(), PcuExecutionFault> {
    observed(plan.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 1), input),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 0), output),
    ]))
}

fn width<T: Format>(backend: &PcuVulkanBackend) {
    let sentinel = T::from(T::ONE + 1);
    for uf in UF {
        for range in [Range::Reject, Range::Clamp] {
            let mut prepared = source_plan::<T, 65>(backend, uf, range);
            let mut graph = graph_plan::<T, 65>(backend, uf, range);
            assert_eq!(graph.profile().resources().len(), 2);
            let mut input = [T::from(T::ONE); 65];
            let mut output = [sentinel; 68];
            let mut graph_output = output;
            for phase in 0..3 {
                for (lane, value) in input.iter_mut().enumerate() {
                    *value = oracle::dyadic::<T>(phase + lane).0;
                }
                prepared(&mut output, &input).unwrap();
                explicit(&mut graph, &input, &mut graph_output).unwrap();
                for (lane, value) in output[..65].iter().enumerate() {
                    assert_eq!(value.bits(), oracle::dyadic::<T>(phase + lane).1.bits());
                }
                assert_eq!(output.map(T::bits), graph_output.map(T::bits));
                assert!(
                    output[65..]
                        .iter()
                        .all(|value| value.bits() == sentinel.bits())
                );
            }
            let old = output.map(T::bits);
            input[2] = T::from(T::SIGN - 1);
            let result = observed(prepared(&mut output, &input));
            assert_eq!(
                result,
                Err(PcuExecutionFault {
                    invocation_id: 2,
                    kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                    recovered: false
                })
            );
            assert_eq!(explicit(&mut graph, &input, &mut graph_output), result);
            assert_eq!(output.map(T::bits), old);
            assert_eq!(graph_output.map(T::bits), old);
            input[2] = T::from(T::ONE);
            assert!(prepared(&mut output[..64], &input).is_err());
            assert_eq!(output.map(T::bits), old);
            input.fill(T::from(T::ONE));
            input[0] = T::from(T::MAX);
            let result = observed(prepared(&mut output, &input));
            assert_eq!(
                result,
                Err(PcuExecutionFault {
                    invocation_id: 0,
                    kind: PcuExecutionFaultKind::ArithmeticOverflow,
                    recovered: range == Range::Clamp
                })
            );
            assert_eq!(explicit(&mut graph, &input, &mut graph_output), result);
            if range == Range::Clamp {
                assert_eq!(output[0].bits(), T::MAX);
                let old = output.map(T::bits);
                input[2] = T::from(T::SIGN - 1);
                assert_eq!(
                    observed(prepared(&mut output, &input)),
                    Err(PcuExecutionFault {
                        invocation_id: 2,
                        kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                        recovered: false
                    })
                );
                assert_eq!(output.map(T::bits), old);
            } else {
                assert_eq!(output.map(T::bits), old);
            }
            input.fill(T::from(T::ONE));
            prepared(&mut output, &input).unwrap();
            assert!(
                output[..65]
                    .iter()
                    .all(|value| value.bits() == T::ONE + T::MIN_NORMAL)
            );
            assert!(
                output[65..]
                    .iter()
                    .all(|value| value.bits() == sentinel.bits())
            );
        }
    }
    grid_and_broadcast::<T>(backend);
    ordered::<T>(backend);
}

fn grid_and_broadcast<T: Format>(backend: &PcuVulkanBackend) {
    let sentinel = T::from(T::ONE + 1);
    let input = [T::from(T::ONE); 65];
    let mut output = [sentinel; 68];
    let mut grid = source::grid_prepare::<T, 65, _>(backend).unwrap();
    grid(&[], &mut output, &input).unwrap();
    let mut broadcast = source::broadcast_prepare::<T, 65, _>(backend).unwrap();
    broadcast(&[], &mut output, &input[0]).unwrap();
    assert!(
        output[..65]
            .iter()
            .all(|value| value.bits() == T::ONE + T::MIN_NORMAL)
    );
    assert!(
        output[65..]
            .iter()
            .all(|value| value.bits() == sentinel.bits())
    );
}

fn ordered<T: Format>(backend: &PcuVulkanBackend) {
    let sentinel = T::from(T::ONE + 1);
    let mut input = [T::from(T::ONE); 7];
    let mut stage = [sentinel; 10];
    let mut output = stage;
    let mut ordered = source::ordered_prepare::<T, 7, _>(backend).unwrap();
    ordered(&input, &mut stage, &mut output).unwrap();
    assert!(
        stage[..7]
            .iter()
            .all(|value| value.bits() == T::ONE + T::MIN_NORMAL)
    );
    assert!(
        output[..7]
            .iter()
            .all(|value| value.bits() == T::ONE + T::MIN_NORMAL)
    );
    let old = stage.map(T::bits);
    input[6] = T::from(T::SIGN - 1);
    assert!(ordered(&input, &mut stage, &mut output).is_err());
    assert_eq!(stage.map(T::bits), old);
    assert_eq!(output.map(T::bits), old);
    input.fill(T::from(1));
    let mut clamp = source::ordered_clamp_prepare::<T, 7, _>(backend).unwrap();
    let fault = observed(clamp(&input, &mut stage, &mut output)).unwrap_err();
    assert_eq!(
        fault,
        PcuExecutionFault {
            invocation_id: 0,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            recovered: true
        }
    );
    assert!(stage[..7].iter().all(|value| value.bits() == 2));
    assert!(output[..7].iter().all(|value| value.bits() == 0));
    let old_stage = stage.map(T::bits);
    let old_output = output.map(T::bits);
    input[6] = T::from(T::SIGN - 1);
    assert_eq!(
        observed(clamp(&input, &mut stage, &mut output)).unwrap_err(),
        PcuExecutionFault {
            invocation_id: 6,
            kind: PcuExecutionFaultKind::InvalidFloatingOperand,
            recovered: false
        }
    );
    assert_eq!(stage.map(T::bits), old_stage);
    assert_eq!(output.map(T::bits), old_output);
    assert!(
        stage[7..]
            .iter()
            .all(|value| value.bits() == sentinel.bits())
    );
    assert!(
        output[7..]
            .iter()
            .all(|value| value.bits() == sentinel.bits())
    );
    input.fill(T::from(T::ONE));
    let mut unused = source::unused_fault_prepare::<T, 7, _>(backend).unwrap();
    let mut divisor = input;
    divisor[2] = T::from(0);
    assert_eq!(
        observed(unused(&input, &divisor, &mut output)).unwrap_err(),
        PcuExecutionFault {
            invocation_id: 2,
            kind: PcuExecutionFaultKind::DivideByZero,
            recovered: false
        }
    );
    assert_eq!(output.map(T::bits), old_output);
    divisor[2] = T::from(T::ONE);
    unused(&input, &divisor, &mut output).unwrap();
}

#[test]
#[ignore = "requires actual Vulkan hardware; ordinary composed source and private transactions"]
fn six_format_source_graph_private_transactions_and_ordered_stores() {
    let backend = selected_template_source();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                let mut policy = global::PcuExecutionPolicy {
                    numerical_mode: mode,
                    ..global::PcuExecutionPolicy::default()
                };
                policy.numerical_options.compound_arithmetic = compound;
                policy.numerical_options.precision = precision;
                global::configure(policy).unwrap();
                width::<pcu_facade::PcuF16Bits>(&backend);
                width::<pcu_facade::PcuBf16Bits>(&backend);
                width::<pcu_facade::PcuF8E4M3FnBits>(&backend);
                width::<pcu_facade::PcuF8E5M2Bits>(&backend);
                width::<f32>(&backend);
                width::<f64>(&backend);
            }
        }
    }
    global::use_defaults().unwrap();
}

#[test]
#[ignore = "requires actual Vulkan hardware; bounded smoke before full encoding proof"]
fn small_static_native_source_smoke() {
    global::use_defaults().unwrap();
    let backend = selected_template_source();
    width::<pcu_facade::PcuF8E4M3FnBits>(&backend);
    width::<f32>(&backend);
    width::<f64>(&backend);
}

fn selected_template_source() -> PcuVulkanBackend {
    let backend = PcuVulkanBackend::new().unwrap();
    if let Some(directory) = std::env::var_os("PCU_COMPOSED_PACKAGES") {
        backend.configure_shader_source(
            fusion_pcu_vulkan::PcuVulkanShaderSource::ExternalComposed {
                directory: directory.into(),
                retain_in_memory: true,
            },
        );
    }
    backend
}

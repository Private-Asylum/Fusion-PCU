//! Ambient flush controls must not change exact checked derivative selection.
#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
extern crate pcu_facade as fusion_pcu;
#[path = "environment/environment.rs"]
mod environment;
#[path = "../../../rocm/benches/checked_relu_backward/oracle/oracle.rs"]
#[allow(dead_code)]
mod oracle;
#[path = "../low_tensor_backward/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use pcu_facade::{global,PcuFloatUnderflowPolicy,PcuExecutionFaultKind,PcuNumericalMode,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,TensorElement,TensorError};
use fusion_pcu_cpu::PcuCpuPreparedTensorGraph;
use oracle::Format;

fn inspect<T: Format>(
    label: &str,
    actual: Result<T, PcuExecutionFaultKind>,
    expected: Result<T, PcuExecutionFaultKind>,
    failures: &mut usize,
) {
    let actual = actual.map(T::bits);
    let expected = expected.map(T::bits);
    if actual != expected {
        *failures += 1;
        eprintln!(
            "environment mismatch {} {label}: actual={actual:?} expected={expected:?}",
            T::LABEL
        );
    }
}

#[allow(clippy::too_many_lines)] // Runtime bit oracle, three actual routes, private fault/tail publication and retry remain adjacent.
fn verify<T: Format + TensorElement>(flush: bool) -> usize {
    let mut failures = 0;
    let cases = [
        (1, T::ONE),
        (T::MIN_NORMAL - 1, T::SIGN | T::ONE),
        (T::SIGN | 1, T::ONE),
        (1, 1),
        (T::SIGN | 1, 1),
        (0, T::SIGN),
        (T::SIGN, T::SIGN),
        (T::ONE, T::SIGN),
        (T::ONE, T::ONE),
        (T::MAX + 1, T::ONE),
        (T::SIGN | (T::MAX + 1), T::ONE),
        (0, T::MAX + 1),
        (T::SIGN | T::ONE, T::MAX + 1),
        (T::ONE, T::MAX + 1),
        (T::ONE, T::SIGN | (T::MAX + 1)),
        (T::SIGN - 1, T::ONE),
        (0, T::SIGN - 1),
        (T::SIGN | T::ONE, T::SIGN - 1),
    ];
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Cpu,
            float_underflow: policy,
            ..Default::default()
        })
        .unwrap();
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let input = graph.input([1], T::TYPE).unwrap();
        let upstream = graph.input([1], T::TYPE).unwrap();
        let derivative = graph.relu_backward(input, upstream).unwrap();
        graph
            .set_value_float_underflow_policy(derivative, policy)
            .unwrap();
        let mut plan = PcuCpuPreparedTensorGraph::<T>::prepare(&graph, &[derivative]).unwrap();
        drop(graph);
        for (input_bits, upstream_bits) in cases {
            let input = std::hint::black_box(T::from(std::hint::black_box(input_bits)));
            let upstream = std::hint::black_box(T::from(std::hint::black_box(upstream_bits)));
            let expected = oracle::expected(input, upstream, policy);
            let label =
                format!("flush={flush} {policy:?} input={input_bits:x} upstream={upstream_bits:x}");
            inspect(
                &format!("core {label}"),
                input.pcu_checked_relu_backward_with_policy(upstream, policy),
                expected,
                &mut failures,
            );
            let sentinel = T::sentinel();
            let mut output = [sentinel; 3];
            let prepared = match plan.call(&[&[input], &[upstream]], &mut [&mut output]) {
                Ok(()) => Ok(output[0]),
                Err(TensorError::ArithmeticFault {
                    element_index: 0,
                    kind,
                    ..
                }) => Err(kind),
                other => panic!("unexpected prepared result {other:?}"),
            };
            inspect(
                &format!("prepared {label}"),
                prepared,
                expected,
                &mut failures,
            );
            assert_eq!(output[1].bits(), sentinel.bits());
            assert_eq!(output[2].bits(), sentinel.bits());
            if prepared.is_err() {
                assert_eq!(output[0].bits(), sentinel.bits());
                assert!(plan.output(0).is_none());
            }
            let ordinary = match source::checked(&[input], &[upstream]) {
                Ok(owner) => {
                    let mut data = [sentinel; 3];
                    owner.read_into(&mut data).unwrap();
                    assert_eq!(data[1].bits(), sentinel.bits());
                    assert_eq!(data[2].bits(), sentinel.bits());
                    Ok(data[0])
                }
                Err(error) => {
                    let fault = error.arithmetic_fault().unwrap();
                    assert_eq!(fault.invocation_id, 0);
                    assert!(!fault.recovered);
                    Err(fault.kind)
                }
            };
            inspect(
                &format!("source {label}"),
                ordinary,
                expected,
                &mut failures,
            );
        }
        let mut output = [T::sentinel(); 3];
        plan.call(&[&[T::one()], &[T::one()]], &mut [&mut output])
            .unwrap();
        assert_eq!(output[0].bits(), T::ONE);
    }
    failures
}

#[test]
fn exact_derivative_is_independent_of_thread_flush_controls() {
    let initial = environment::state();
    let mut failures = 0;
    for flush in [false, true] {
        {
            let _restore = environment::Guard::enter(flush);
            eprintln!("thread flush={flush} state={:?}", environment::state());
            failures += verify::<f32>(flush);
            failures += verify::<f64>(flush);
            failures += verify::<PcuF16Bits>(flush);
            failures += verify::<PcuBf16Bits>(flush);
            failures += verify::<PcuF8E4M3FnBits>(flush);
            failures += verify::<PcuF8E5M2Bits>(flush);
        }
        assert_eq!(
            environment::state(),
            initial,
            "RAII restores exact control/status even across failed arithmetic"
        );
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    assert_eq!(
        failures, 0,
        "runtime core/prepared/genuine-source selection mismatches"
    );
}

//! Actual ordinary source, detached prepared source and independent exact-bit models.
extern crate pcu_facade as fusion_pcu;
#[path = "graph/graph.rs"]
mod graph;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCompoundArithmeticPolicy,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuBindingRef,
    PcuHostArgument,
    PcuPreparedHostKernel,
};
use fusion_pcu_cpu::PcuCpuHostBackend;
type Prepared<T> = Box<dyn FnMut(&mut [T], &[T]) -> Result<(), fusion_pcu_cpu::PcuCpuHostError>>;
fn prepared<T: oracle::Format, const N: usize>(
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Prepared<T> {
    let backend = PcuCpuHostBackend::scalar();
    match (policy, range) {
        (PcuFloatUnderflowPolicy::IeeeAfterRounding, PcuRangePolicy::Reject) => {
            Box::new(source::reject_ieee_prepare::<T, N, _>(&backend).unwrap())
        }
        (PcuFloatUnderflowPolicy::AllowGradualUnderflow, PcuRangePolicy::Reject) => {
            Box::new(source::reject_gradual_prepare::<T, N, _>(&backend).unwrap())
        }
        (PcuFloatUnderflowPolicy::RejectSubnormalResult, PcuRangePolicy::Reject) => {
            Box::new(source::reject_tight_prepare::<T, N, _>(&backend).unwrap())
        }
        (PcuFloatUnderflowPolicy::IeeeAfterRounding, PcuRangePolicy::Clamp) => {
            Box::new(source::clamp_ieee_prepare::<T, N, _>(&backend).unwrap())
        }
        (PcuFloatUnderflowPolicy::AllowGradualUnderflow, PcuRangePolicy::Clamp) => {
            Box::new(source::clamp_gradual_prepare::<T, N, _>(&backend).unwrap())
        }
        (PcuFloatUnderflowPolicy::RejectSubnormalResult, PcuRangePolicy::Clamp) => {
            Box::new(source::clamp_tight_prepare::<T, N, _>(&backend).unwrap())
        }
    }
}
fn low_width<T: oracle::Format>() {
    let limit = T::SIGN * 2;
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let mut execute = prepared::<T, 1>(policy, range);
            let mut graph = graph::prepare::<T, 1>(policy, range);
            for raw in 0..limit {
                let value = T::from(raw);
                let sentinel = T::from(T::ONE + 1);
                let mut output = [sentinel; 3];
                let (fault, expected) = oracle::low_expected(value, policy, range);
                let actual = execute(&mut output, &[value]).map_err(|error| error.fault().unwrap());
                assert_eq!(
                    actual,
                    fault.map_or(Ok(()), Err),
                    "{:?} raw{raw:x} {policy:?}/{range:?}",
                    T::TYPE
                );
                let expected = expected.unwrap_or(sentinel);
                assert_eq!(output[0].bits(), expected.bits());
                assert_eq!(output[1].bits(), sentinel.bits());
                assert_eq!(output[2].bits(), sentinel.bits());
                let mut graph_output = [sentinel; 3];
                let graph_result = graph
                    .call(&mut [
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut graph_output),
                        PcuHostArgument::read(PcuBindingRef::new(0, 1), &[value]),
                    ])
                    .map_err(|error| error.fault().unwrap());
                assert_eq!(graph_result, actual);
                assert_eq!(graph_output.map(T::bits), output.map(T::bits));
            }
        }
    }
}
#[test]
fn complete_low_encodings_preserve_per_step_faults_payload_and_tails() {
    low_width::<pcu_facade::PcuF16Bits>();
    low_width::<pcu_facade::PcuBf16Bits>();
    low_width::<pcu_facade::PcuF8E4M3FnBits>();
    low_width::<pcu_facade::PcuF8E5M2Bits>();
}
fn source_width<T: oracle::Format>() {
    let sentinel = T::from(T::ONE + 1);
    let mut output = [sentinel; 9];
    let mut input = [T::from(T::ONE); 6];
    for phase in 0..3 {
        for (lane, value) in input.iter_mut().enumerate() {
            *value = oracle::dyadic::<T>(lane + phase).0;
        }
        oracle::dyadic_native(&input, &mut output, 6).unwrap();
        source::reject_ieee::<T, 6>(&mut output, &input).unwrap();
        for (lane, value) in output[..6].iter().enumerate() {
            assert_eq!(value.bits(), oracle::dyadic::<T>(lane + phase).1.bits());
        }
        source::grid::<T, 6>(&[], &mut output, &input).unwrap();
        source::broadcast::<T, 6>(&[], &mut output, &input[0]).unwrap();
        for value in &output[..6] {
            assert_eq!(value.bits(), oracle::dyadic::<T>(phase).1.bits());
        }
        for value in &output[6..] {
            assert_eq!(value.bits(), sentinel.bits());
        }
    }
    let old = output.map(T::bits);
    input[2] = T::from(T::SIGN - 1);
    let error = source::reject_ieee::<T, 6>(&mut output, &input).unwrap_err();
    assert_eq!(
        error.arithmetic_fault().unwrap(),
        PcuExecutionFault {
            invocation_id: 2,
            kind: pcu_facade::PcuExecutionFaultKind::InvalidFloatingOperand,
            recovered: false
        }
    );
    assert_eq!(output.map(T::bits), old);
    input[2] = T::from(T::ONE);
    assert!(source::reject_ieee::<T, 6>(&mut output[..5], &input).is_err());
    assert_eq!(output.map(T::bits), old);
    source::reject_ieee::<T, 6>(&mut output, &input).unwrap();
    ordinary_recovery::<T>();
}
fn ordinary_recovery<T: oracle::Format>() {
    for entry in [
        source::clamp_ieee::<T, 4>,
        source::clamp_gradual::<T, 4>,
        source::clamp_tight::<T, 4>,
    ] {
        let one = T::from(T::ONE);
        let mut input = [T::from(T::MAX), one, one, one];
        let mut output = [T::from(T::ONE + 1); 7];
        let fault = entry(&mut output, &input)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            fault,
            PcuExecutionFault {
                invocation_id: 0,
                kind: pcu_facade::PcuExecutionFaultKind::ArithmeticOverflow,
                recovered: true
            }
        );
        assert_eq!(output[0].bits(), T::MAX);
        let old = output.map(T::bits);
        input[2] = T::from(T::SIGN - 1);
        let fault = entry(&mut output, &input)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            fault,
            PcuExecutionFault {
                invocation_id: 2,
                kind: pcu_facade::PcuExecutionFaultKind::InvalidFloatingOperand,
                recovered: false
            }
        );
        assert_eq!(output.map(T::bits), old);
        input[0] = one;
        input[2] = one;
        entry(&mut output, &input).unwrap();
        for value in &output[..4] {
            assert_eq!(value.bits(), T::ONE + T::MIN_NORMAL);
        }
        for value in &output[4..] {
            assert_eq!(value.bits(), T::ONE + 1);
        }
    }
}
#[test]
fn six_formats_genuine_ordinary_grid_broadcast_and_host_transaction() {
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
                    backend: global::PcuBackendChoice::Cpu,
                    ..Default::default()
                };
                policy.numerical_mode = mode;
                policy.numerical_options.compound_arithmetic = compound;
                policy.numerical_options.precision = precision;
                global::configure(policy).unwrap();
                source_width::<pcu_facade::PcuF16Bits>();
                source_width::<pcu_facade::PcuBf16Bits>();
                source_width::<pcu_facade::PcuF8E4M3FnBits>();
                source_width::<pcu_facade::PcuF8E5M2Bits>();
                source_width::<f32>();
                source_width::<f64>();
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

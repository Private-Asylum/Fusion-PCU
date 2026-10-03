//! N=1 native private-word reload; larger cross-lane requests remain cold refusals.
use super::*;

fn width<T: Format>(backend: &PcuVulkanBackend) {
    let sentinel = T::from(T::ONE + 1);
    let mut stage = [sentinel; 4];
    let mut output = stage;
    let mut execute = source::element_zero_after_store_prepare::<T, 1, _>(backend).unwrap();
    for phase in 0..3 {
        let (input, expected) = oracle::dyadic::<T>(phase);
        execute(&[input], &mut stage, &mut output).unwrap();
        assert_eq!(output[0].bits(), expected.bits());
        assert!(
            stage[1..]
                .iter()
                .all(|value| value.bits() == sentinel.bits())
        );
        assert!(
            output[1..]
                .iter()
                .all(|value| value.bits() == sentinel.bits())
        );
    }
    let old_stage = stage.map(T::bits);
    let old_output = output.map(T::bits);
    let invalid = T::from(T::SIGN - 1);
    assert_eq!(
        observed(execute(&[invalid], &mut stage, &mut output)),
        Err(PcuExecutionFault {
            invocation_id: 0,
            kind: PcuExecutionFaultKind::InvalidFloatingOperand,
            recovered: false,
        })
    );
    assert_eq!(stage.map(T::bits), old_stage);
    assert_eq!(output.map(T::bits), old_output);
    execute(&[T::from(T::ONE)], &mut stage, &mut output).unwrap();
    assert_eq!(stage[0].bits(), T::ONE + T::MIN_NORMAL);
    assert_eq!(output[0].bits(), stage[0].bits());
    assert!(source::element_zero_after_store_prepare::<T, 7, _>(backend).is_err());
}

#[test]
#[ignore = "requires Vulkan hardware; corrected cold N=1 private-shadow specialization"]
fn single_lane_element_zero_reload_uses_the_ordered_private_store() {
    let backend = PcuVulkanBackend::new().unwrap();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for uf in UF {
                    for range in [Range::Reject, Range::Clamp] {
                        let mut policy = global::PcuExecutionPolicy {
                            numerical_mode: mode,
                            float_underflow: uf,
                            range_policy: range,
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
        }
    }
    global::use_defaults().unwrap();
}

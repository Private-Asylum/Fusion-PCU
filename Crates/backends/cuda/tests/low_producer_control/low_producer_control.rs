//! Frozen handwritten CUDA arithmetic is independently cross-checked with PCU primitive laws.
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use fusion_pcu::{
    PcuFloatUnderflowPolicy,
    PcuExecutionFaultKind,
};
#[path = "../../benches/low_tensor_producers/oracle/oracle.rs"]
#[allow(dead_code)]
// The paired benchmark exercises all oracle helpers; this test only checks fixed vectors.
mod oracle;
use oracle::Format;
fn code(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 3,
        PcuExecutionFaultKind::ArithmeticUnderflow => 4,
        PcuExecutionFaultKind::InvalidFloatingOperand => 5,
        _ => panic!("unsupported fault"),
    }
}
fn check<T: Format>() {
    for row in include_str!("../../benches/low_tensor_producers/oracle/vectors.txt")
        .lines()
        .filter(|row| !row.starts_with('#'))
    {
        let fields = row.split_whitespace().collect::<Vec<_>>();
        if fields[0] != T::LABEL {
            continue;
        }
        let policy = match fields[1] {
            "0" => PcuFloatUnderflowPolicy::IeeeAfterRounding,
            "1" => PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            "2" => PcuFloatUnderflowPolicy::RejectSubnormalResult,
            _ => panic!("invalid policy"),
        };
        let multiply = fields[2] == "1";
        let a = T::from(u16::from_str_radix(fields[3], 16).unwrap());
        let b = T::from(u16::from_str_radix(fields[4], 16).unwrap());
        let bits = u16::from_str_radix(fields[5], 16).unwrap();
        let fault_code = fields[6].parse::<u32>().unwrap();
        let expected = if fault_code == 0 {
            Ok(T::from(bits))
        } else {
            Err(fault_code)
        };
        let observed = if multiply {
            a.pcu_clamped_mul_with_policy(b, policy)
        } else {
            a.pcu_clamped_add_with_policy(b, policy)
        }
        .map_err(|error| code(error.kind()));
        assert_eq!(observed, expected, "{row}");
    }
}
#[test]
fn frozen_handwritten_low_control_matches_all_four_primitive_laws_and_policies() {
    check::<fusion_pcu::PcuF16Bits>();
    check::<fusion_pcu::PcuBf16Bits>();
    check::<fusion_pcu::PcuF8E4M3FnBits>();
    check::<fusion_pcu::PcuF8E5M2Bits>();
}

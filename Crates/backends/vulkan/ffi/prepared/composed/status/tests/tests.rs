//! Synthetic protocol exercises: no injected driver failure or native execution claim.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchFloatBinaryOp as Op,
    PcuFloatUnderflowPolicy as Uf,
    PcuRangePolicy as Range,
    PcuScalarType as Scalar,
};

fn law(step: usize) -> Option<PcuCheckedScalarFaultLaw> {
    match step {
        1 => PcuCheckedScalarFaultLaw::float_binary(
            Scalar::F32,
            Op::Add,
            Range::Clamp,
            Uf::IeeeAfterRounding,
        ),
        3 => PcuCheckedScalarFaultLaw::float_binary(
            Scalar::F32,
            Op::Mul,
            Range::Reject,
            Uf::RejectSubnormalResult,
        ),
        5 => PcuCheckedScalarFaultLaw::float_binary(
            Scalar::F32,
            Op::Div,
            Range::Clamp,
            Uf::AllowGradualUnderflow,
        ),
        _ => None,
    }
}

#[test]
fn exact_step_law_excludes_union_only_and_invalid_protocol_records() {
    for record in [
        (0, 1),
        (1, 0),
        (1, 99),
        (2, 1),
        (0x103, 3),
        (3, 1),
        (0x101, 1),
        (0x102, 5),
        (4, 3),
        (5, 5),
        (0x203, 1),
        (0xffff_ffff, 1),
    ] {
        assert!(
            matches!(
                scan_records(1, law, |_| record),
                Err(PcuVulkanError::InvalidStatus { .. })
            ),
            "record {record:?}"
        );
    }
    assert_eq!(scan_records(1, law, |_| (0, 0)).unwrap(), None);
}

#[test]
fn fatal_priority_and_full_span_protocol_validation_precede_publication() {
    let records = [(0x103, 1), (0, 0), (2, 3), (4, 5), (0x103, 1)];
    let Err(PcuVulkanError::Fault(fault)) = scan_records(5, law, |lane| records[lane]) else {
        panic!("first fatal was not selected");
    };
    assert_eq!(fault.invocation_id, 2);
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
    assert!(!fault.recovered);
    assert!(matches!(
        scan_records(6, law, |lane| if lane < 5 {
            records[lane]
        } else {
            (3, 99)
        }),
        Err(PcuVulkanError::InvalidStatus {
            invocation_id: 5,
            ..
        })
    ));
    let notice = scan_records(3, law, |lane| if lane == 1 { (0x103, 1) } else { (0, 0) })
        .unwrap()
        .unwrap();
    assert_eq!(notice.invocation_id, 1);
    assert!(notice.recovered);
}

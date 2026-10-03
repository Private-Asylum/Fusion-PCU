#[rustfmt::skip]
use crate::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
};

const fn record(invocation_id: u64, recovered: bool) -> PcuExecutionFault {
    PcuExecutionFault {
        kind: PcuExecutionFaultKind::ArithmeticOverflow,
        invocation_id,
        recovered,
    }
}

#[test]
fn decoded_status_needs_the_prepared_extent_even_with_a_legal_physical_index() {
    for recovered in [false, true] {
        assert!(!record(8, recovered).is_within_logical_extent(4));
        assert!(!record(4, recovered).is_within_logical_extent(4));
        assert!(record(3, recovered).is_within_logical_extent(4));
        assert!(!record(0, recovered).is_within_logical_extent(0));
    }
}

#[test]
fn grid_stride_fault_domain_is_not_the_launch_invocation_count() {
    let fault = record(8, false);
    assert!(!fault.is_within_logical_extent(3));
    assert!(fault.is_within_logical_extent(19));
}

#[test]
fn neutral_fault_domains_are_not_limited_to_a_gpu_u32_status_profile() {
    let u32_last = record(u32::MAX.into(), false);
    assert!(u32_last.is_within_logical_extent(1_u64 << 32));
    let above_u32 = record(1_u64 << 32, false);
    assert!(above_u32.is_within_logical_extent((1_u64 << 32) + 1));
    assert!(record(u64::MAX - 1, false).is_within_logical_extent(u64::MAX));
    assert!(!record(u64::MAX, false).is_within_logical_extent(u64::MAX));
}

//! Actual annotated source against the explicit Metal host preparation seam.

use pcu_facade::pcu;
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalError,
    MetalSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuExecutionFaultKind,
    PcuHostDispatchError,
};

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn checked_add<const N: usize>(lhs: &[u32], rhs: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] + rhs[id];
}

#[test]
#[ignore = "Requires actual macOS Metal device and runs annotated source on GPU."]
fn annotated_source_success_fault_retry_and_tail_preservation() {
    let session = MetalSession::open(0).unwrap();
    let mut source = checked_add_prepare::<3, _>(&session).unwrap();
    let mut output = [91_u32; 5];
    source(&[1, 2, 3], &[3, 4, 5], &mut output).unwrap();
    assert_eq!(output, [4, 6, 8, 91, 91]);
    let before = output;
    assert!(
        matches!(source(&[1, u32::MAX, u32::MAX], &[3, 1, 1], &mut output), Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id == 1 && !fault.recovered)
    );
    assert_eq!(output, before);
    source(&[7, 8, 9], &[1, 2, 3], &mut output).unwrap();
    assert_eq!(output, [8, 10, 12, 91, 91]);
    assert!(matches!(
        source(&[1], &[1, 2, 3], &mut output),
        Err(PcuHostDispatchError::BufferTooSmall(_))
    ));
    assert_eq!(output, [8, 10, 12, 91, 91]);
    drop(session);
    source(&[2, 2, 2], &[3, 3, 3], &mut output).unwrap();
    assert_eq!(output, [5, 5, 5, 91, 91]);
}

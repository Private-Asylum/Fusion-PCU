//! A terminal failed mutable resident write is either unsubmitted or discarded.
#[rustfmt::skip]
use fusion_pcu::{
    PcuArgumentError,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

/// A backend that commits only after validation can preserve the old value.
/// A possible-write fatal result must block reads. Uncertain completion or any
/// unrelated failure cannot masquerade as either of these terminal outcomes.
#[track_caller]
pub fn discarded_after_terminal_fault<T: PcuScalar>(
    value: &PcuTensor<T>,
    host: &mut [T],
    assert_unchanged: impl FnOnce(&[T], &[T]),
) -> bool {
    let before = host.to_vec();
    match value.read_into(host) {
        Ok(()) => false,
        Err(PcuExecutionError::Argument(PcuArgumentError::ResidentValueDiscarded)) => {
            assert_unchanged(host, &before);
            true
        }
        Err(error) => {
            panic!("unexpected resident state after a terminal arithmetic fault: {error:?}")
        }
    }
}

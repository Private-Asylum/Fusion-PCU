//! Shared bit comparison and process-global policy serialization for all parity families.
use std::sync::Mutex;
use fusion_pcu::PcuScalar;

pub static POLICY_LOCK: Mutex<()> = Mutex::new(());

pub fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}

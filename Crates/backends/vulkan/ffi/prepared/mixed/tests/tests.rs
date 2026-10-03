use super::*;
#[rustfmt::skip]
use crate::ffi::{
    VulkanDevice,
    VulkanOwnedBuffer,
};
use std::rc::Rc;
#[path = "../../../../../spirv/scalar_transport/bytecode/bytecode.rs"]
#[allow(clippy::redundant_pub_crate)]
// Production template is private to its own parent; test only includes its immutable words.
mod bytecode;
#[test]
#[ignore = "requires actual Vulkan device; deterministic protocol witness, not injected driver failure"]
fn uncertain_protocol_quarantines_inputs_outputs_and_retains_execution_roots() {
    for public_commit in [false, true] {
        let device = Rc::new(VulkanDevice::new().unwrap());
        let mut plan = VulkanPreparedMap::<2, false>::new(
            Rc::clone(&device),
            bytecode::WORDS,
            1,
            1,
            [64, 1, 1],
            [4, 4],
            StatusPolicy::ZERO_ONLY,
        )
        .unwrap();
        plan.enable_mixed().unwrap();
        let input = VulkanOwnedBuffer::new(&device, 4).unwrap();
        let output = VulkanOwnedBuffer::new(&device, 4).unwrap();
        let references = Rc::strong_count(&device);
        let mut state = WriteState {
            may_have_written: public_commit,
            completion_uncertain: false,
        };
        // Exercise the exact native completion disposition without manufacturing a driver
        // failure or submitting unfinished work. The original source cause remains distinct.
        plan.quarantine(&mut state);
        assert_eq!(state.may_have_written, public_commit);
        assert!(state.completion_uncertain);
        assert!(plan.resources.is_none());
        assert!(plan.commit.is_some());
        assert_eq!(Rc::strong_count(&device), references + 1);
        assert!(matches!(
            input.validate_access_available(),
            Err(PcuVulkanError::Quarantined)
        ));
        assert!(matches!(
            output.validate_access_available(),
            Err(PcuVulkanError::Quarantined)
        ));
        drop((input, output, plan));
        assert!(Rc::strong_count(&device) > 1); // Full actual device/queue root remains retained.
    }
}

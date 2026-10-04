//! Completion-result injection exercises actual retained owners without device-loss fiction.
#[rustfmt::skip]
use crate::{
    HipError,
    HipRuntime,
};
use std::rc::Rc;

fn error(operation: &'static str) -> HipError {
    HipError::Runtime {
        operation,
        code: -1,
        detail: None,
    }
}

#[test]
#[ignore = "requires HIP device; known-terminal copy error precedence and access release"]
fn known_terminal_releases_all_access_and_preserves_copy_error() {
    let runtime = HipRuntime::new(0).unwrap();
    let source = runtime.allocate(16).unwrap();
    let destination = runtime.allocate(16).unwrap();
    for copy in [Ok(()), Err(error("copy"))] {
        let source_lease = source.acquire_access().unwrap();
        let destination_lease = destination.acquire_access().unwrap();
        let expected = copy.clone();
        assert_eq!(
            super::finish_copy([source_lease, destination_lease], copy, Ok(())),
            expected
        );
        assert!(source.validate_access_available().is_ok());
        assert!(destination.validate_access_available().is_ok());
    }
}

#[test]
#[ignore = "requires HIP device; uncertain completion retains both endpoints and first error"]
fn unknown_terminal_retains_all_endpoints_and_preserves_first_error() {
    let runtime = HipRuntime::new(0).unwrap();
    for copy in [Ok(()), Err(error("copy"))] {
        let source = runtime.allocate(16).unwrap();
        let destination = runtime.allocate(16).unwrap();
        let source_owner = Rc::downgrade(&source.allocation);
        let destination_owner = Rc::downgrade(&destination.allocation);
        let source_lease = source.acquire_access().unwrap();
        let destination_lease = destination.acquire_access().unwrap();
        // These valid allocations have no queued work. Inject only the completion
        // result through the same retention branch as a failed actual stream wait.
        // No invalid native handle is passed and no hardware device loss is claimed.
        let expected = copy.clone().and_then(|()| Err(error("completion")));
        assert_eq!(
            super::finish_copy(
                [source_lease, destination_lease],
                copy,
                Err(error("completion"))
            ),
            expected
        );
        assert!(source.validate_access_available().is_err());
        assert!(destination.validate_access_available().is_err());
        drop(source);
        drop(destination);
        assert!(source_owner.upgrade().is_some());
        assert!(destination_owner.upgrade().is_some());
    }
}

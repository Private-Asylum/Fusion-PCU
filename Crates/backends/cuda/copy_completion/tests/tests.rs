#[rustfmt::skip]
use crate::{
    CudaError,
    CudaRuntime,
};
use std::rc::Rc;

fn error(operation: &'static str) -> CudaError {
    CudaError::Runtime {
        operation,
        code: -1,
        detail: None,
    }
}

#[test]
#[ignore = "requires native CUDA; copy error precedence and known-terminal access release"]
fn copy_completion_known_terminal_releases_access_and_preserves_copy_error() {
    let runtime = CudaRuntime::new(0).unwrap();
    let buffer = runtime.allocate(16).unwrap();
    let lease = buffer.acquire_access().unwrap();
    assert_eq!(
        super::finish_copy([lease], Err(error("copy")), Ok(())),
        Err(error("copy"))
    );
    assert!(buffer.validate_access_available().is_ok());
    let lease = buffer.acquire_access().unwrap();
    super::finish_copy([lease], Ok(()), Ok(())).unwrap();
    assert!(buffer.validate_access_available().is_ok());
}

#[test]
#[ignore = "requires native CUDA; uncertain copy completion retains and quarantines both owners"]
fn copy_completion_unknown_terminal_retains_both_endpoints_and_first_error() {
    let runtime = CudaRuntime::new(0).unwrap();
    for copy in [Ok(()), Err(error("copy"))] {
        let source = runtime.allocate(16).unwrap();
        let destination = runtime.allocate(16).unwrap();
        let source_owner = Rc::downgrade(&source.allocation);
        let destination_owner = Rc::downgrade(&destination.allocation);
        let source_lease = source.acquire_access().unwrap();
        let destination_lease = destination.acquire_access().unwrap();
        // Allocations have no queued work. Inject only the owned completion result, with no
        // invalid native handle or device-loss claim. This exercises the exact quarantine
        // branch used after a real default-stream wait failure, including both D2D endpoints.
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

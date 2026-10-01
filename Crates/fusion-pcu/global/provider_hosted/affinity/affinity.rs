//! Source affinity is inhabited only when a provider can own resident source arguments.
//! The host-only build uses an uninhabited type: it cannot fabricate a resident session.

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
pub(super) type ResidentAffinity = std::rc::Rc<super::super::resident::Session>;
#[cfg(not(any(feature = "rocm", feature = "cuda", feature = "metal")))]
#[derive(Clone)]
pub(super) enum ResidentAffinity {}

#[cfg_attr(
    not(any(feature = "rocm", feature = "cuda", feature = "metal")),
    allow(clippy::missing_const_for_fn)
)]
pub(super) fn validate(
    root: &ResidentAffinity,
    policy: super::PcuExecutionPolicy,
) -> Result<(), super::PcuExecutionError> {
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    {
        root.validate_policy(policy)
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "metal")))]
    {
        let _ = (root, policy);
        Err(super::PcuExecutionError::ResidentPolicyConflict)
    }
}

#[cfg_attr(
    not(any(feature = "rocm", feature = "cuda", feature = "metal")),
    allow(clippy::missing_const_for_fn)
)]
pub(super) fn same(left: &ResidentAffinity, right: &ResidentAffinity) -> bool {
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    {
        std::rc::Rc::ptr_eq(left, right)
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "metal")))]
    {
        let _ = (left, right);
        false
    }
}

#[cfg_attr(
    not(any(feature = "rocm", feature = "cuda", feature = "metal")),
    allow(clippy::missing_const_for_fn, clippy::unnecessary_wraps)
)] // The same source seam validates resident sessions when they are available.
pub(super) fn from_arguments<'a>(
    arguments: &[super::super::arguments::PcuCallArgument<'a>],
) -> Result<Option<&'a ResidentAffinity>, super::PcuExecutionError> {
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    {
        let mut affinity = None;
        for argument in arguments {
            let session = match argument.kind() {
                super::super::arguments::PcuCallArgumentKind::Host(_) => continue,
                super::super::arguments::PcuCallArgumentKind::ResidentRead(resident) => {
                    resident.session
                }
                super::super::arguments::PcuCallArgumentKind::ResidentWrite(resident) => {
                    resident.session
                }
            };
            if affinity.is_some_and(|root| !same(root, session)) {
                return Err(super::PcuExecutionError::Argument(
                    super::super::PcuArgumentError::SessionMismatch,
                ));
            }
            affinity = Some(session);
        }
        Ok(affinity)
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "metal")))]
    {
        let _ = arguments;
        Ok(None)
    }
}

//! Cold retained roots and allocation-free borrowed affinity for warm source calls.

#[derive(Clone)]
pub(super) enum ResidentAffinity {
    #[cfg(feature = "cpu")]
    Cpu,
    #[cfg(all(feature = "vulkan", feature = "tensor"))]
    Vulkan(std::rc::Rc<fusion_pcu_vulkan::PcuVulkanBackend>),
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    Device(std::rc::Rc<super::super::resident::Session>),
    #[cfg(feature = "mlx")]
    Mlx(std::rc::Rc<super::super::arguments::MlxSourceRoot>),
    #[cfg(not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        feature = "cpu",
        all(feature = "vulkan", feature = "tensor")
    )))]
    Unavailable,
}
#[derive(Clone, Copy)]
pub(super) enum BorrowedAffinity<'a> {
    #[cfg(feature = "cpu")]
    Cpu(core::marker::PhantomData<&'a ()>),
    #[cfg(all(feature = "vulkan", feature = "tensor"))]
    Vulkan(&'a std::rc::Rc<fusion_pcu_vulkan::PcuVulkanBackend>),
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    Device(&'a std::rc::Rc<super::super::resident::Session>),
    #[cfg(feature = "mlx")]
    Mlx(&'a std::rc::Rc<super::super::arguments::MlxSourceRoot>),
    #[cfg(not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        feature = "cpu",
        all(feature = "vulkan", feature = "tensor")
    )))]
    Unavailable(core::marker::PhantomData<&'a ()>),
}

pub(super) const fn borrow(root: &ResidentAffinity) -> BorrowedAffinity<'_> {
    match root {
        #[cfg(feature = "cpu")]
        ResidentAffinity::Cpu => BorrowedAffinity::Cpu(core::marker::PhantomData),
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        ResidentAffinity::Device(root) => BorrowedAffinity::Device(root),
        #[cfg(all(feature = "vulkan", feature = "tensor"))]
        ResidentAffinity::Vulkan(root) => BorrowedAffinity::Vulkan(root),
        #[cfg(feature = "mlx")]
        ResidentAffinity::Mlx(root) => BorrowedAffinity::Mlx(root),
        #[cfg(not(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "mlx",
            feature = "cpu",
            all(feature = "vulkan", feature = "tensor")
        )))]
        ResidentAffinity::Unavailable => BorrowedAffinity::Unavailable(core::marker::PhantomData),
    }
}

/// Retaining a root is cold preparation only; warm calls borrow the existing owner.
#[cfg_attr(
    not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        feature = "cpu",
        all(feature = "vulkan", feature = "tensor")
    )),
    allow(clippy::missing_const_for_fn)
)] // Runtime provider branches require non-const session checks.
#[cfg_attr(
    all(
        feature = "cpu",
        not(any(feature = "rocm", feature = "cuda", feature = "metal", feature = "mlx"))
    ),
    allow(clippy::missing_const_for_fn)
)] // Other provider branches require runtime retained-session checks.
pub(super) fn retain(root: BorrowedAffinity<'_>) -> ResidentAffinity {
    match root {
        #[cfg(feature = "cpu")]
        BorrowedAffinity::Cpu(_) => ResidentAffinity::Cpu,
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        BorrowedAffinity::Device(root) => ResidentAffinity::Device(std::rc::Rc::clone(root)),
        #[cfg(all(feature = "vulkan", feature = "tensor"))]
        BorrowedAffinity::Vulkan(root) => ResidentAffinity::Vulkan(std::rc::Rc::clone(root)),
        #[cfg(feature = "mlx")]
        BorrowedAffinity::Mlx(root) => ResidentAffinity::Mlx(std::rc::Rc::clone(root)),
        #[cfg(not(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "mlx",
            feature = "cpu",
            all(feature = "vulkan", feature = "tensor")
        )))]
        BorrowedAffinity::Unavailable(_) => ResidentAffinity::Unavailable,
    }
}

#[cfg_attr(
    not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        feature = "cpu",
        all(feature = "vulkan", feature = "tensor")
    )),
    allow(clippy::missing_const_for_fn)
)] // Runtime provider branches require non-const session checks.
pub(super) fn validate(
    root: BorrowedAffinity<'_>,
    policy: super::PcuExecutionPolicy,
) -> Result<(), super::PcuExecutionError> {
    match root {
        #[cfg(feature = "cpu")]
        BorrowedAffinity::Cpu(_) => {
            if !matches!(
                policy.backend,
                super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Cpu
            ) || policy.device.is_some_and(|id| id != 0)
            {
                Err(super::PcuExecutionError::ResidentPolicyConflict)
            } else {
                Ok(())
            }
        }
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        BorrowedAffinity::Device(root) => root.validate_policy(policy),
        #[cfg(all(feature = "vulkan", feature = "tensor"))]
        BorrowedAffinity::Vulkan(root) => {
            if !matches!(
                policy.backend,
                super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Vulkan
            ) || policy.device.is_some_and(|id| {
                root.device_identity()
                    .is_none_or(|identity| identity.device_id() != id)
            }) {
                Err(super::PcuExecutionError::ResidentPolicyConflict)
            } else {
                Ok(())
            }
        }
        #[cfg(feature = "mlx")]
        BorrowedAffinity::Mlx(root) => {
            if !matches!(
                policy.backend,
                super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Mlx
            ) || policy.device.is_some_and(|ordinal| {
                usize::try_from(ordinal).ok() != Some(root.session.facts().index)
            }) {
                Err(super::PcuExecutionError::ResidentPolicyConflict)
            } else {
                Ok(())
            }
        }
        #[cfg(not(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "mlx",
            feature = "cpu",
            all(feature = "vulkan", feature = "tensor")
        )))]
        BorrowedAffinity::Unavailable(_) => {
            let _ = policy;
            Err(super::PcuExecutionError::ResidentPolicyConflict)
        }
    }
}

#[cfg_attr(
    not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        feature = "cpu",
        all(feature = "vulkan", feature = "tensor")
    )),
    allow(clippy::missing_const_for_fn)
)] // Runtime provider branches require non-const session checks.
#[cfg_attr(
    all(
        feature = "cpu",
        not(any(feature = "rocm", feature = "cuda", feature = "metal", feature = "mlx"))
    ),
    allow(clippy::missing_const_for_fn)
)] // Other provider branches require runtime retained-session checks.
pub(super) fn same(left: BorrowedAffinity<'_>, right: BorrowedAffinity<'_>) -> bool {
    match (left, right) {
        #[cfg(all(feature = "vulkan", feature = "tensor"))]
        (BorrowedAffinity::Vulkan(left), BorrowedAffinity::Vulkan(right)) => {
            std::rc::Rc::ptr_eq(left, right)
        }
        #[cfg(all(
            feature = "vulkan",
            feature = "tensor",
            any(
                feature = "cpu",
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                feature = "mlx"
            )
        ))]
        (BorrowedAffinity::Vulkan(_), _) | (_, BorrowedAffinity::Vulkan(_)) => false,
        #[cfg(feature = "cpu")]
        (BorrowedAffinity::Cpu(_), BorrowedAffinity::Cpu(_)) => true,
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        (BorrowedAffinity::Device(left), BorrowedAffinity::Device(right)) => {
            if std::rc::Rc::ptr_eq(left, right) {
                return true;
            }
            #[cfg(feature = "metal")]
            {
                left.shares_metal_session(right)
            }
            #[cfg(not(feature = "metal"))]
            {
                false
            }
        }
        #[cfg(feature = "mlx")]
        (BorrowedAffinity::Mlx(left), BorrowedAffinity::Mlx(right)) => {
            left.session.same_session(&right.session)
        }
        #[cfg(all(
            feature = "cpu",
            any(feature = "rocm", feature = "cuda", feature = "metal", feature = "mlx")
        ))]
        (BorrowedAffinity::Cpu(_), _) | (_, BorrowedAffinity::Cpu(_)) => false,
        #[cfg(all(
            feature = "mlx",
            any(feature = "rocm", feature = "cuda", feature = "metal")
        ))]
        _ => false,
        #[cfg(not(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "mlx",
            feature = "cpu",
            all(feature = "vulkan", feature = "tensor")
        )))]
        (BorrowedAffinity::Unavailable(_), _) => false,
    }
}

#[cfg_attr(
    not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    )),
    allow(clippy::missing_const_for_fn, clippy::unnecessary_wraps)
)]
pub(super) fn from_arguments<'a>(
    arguments: &[super::super::arguments::PcuCallArgument<'a>],
) -> Result<Option<BorrowedAffinity<'a>>, super::PcuExecutionError> {
    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    ))]
    {
        use super::super::arguments::PcuCallArgumentKind;
        let mut affinity = None;
        for argument in arguments {
            let root = match argument.kind() {
                PcuCallArgumentKind::Host(_) => continue,
                #[cfg(all(feature = "vulkan", feature = "tensor"))]
                PcuCallArgumentKind::VulkanRead(resident) => {
                    BorrowedAffinity::Vulkan(resident.root)
                }
                #[cfg(all(feature = "vulkan", feature = "tensor"))]
                PcuCallArgumentKind::VulkanWrite(resident) => {
                    BorrowedAffinity::Vulkan(resident.root)
                }
                #[cfg(all(feature = "cpu", feature = "tensor"))]
                PcuCallArgumentKind::CpuOwner(_) => {
                    BorrowedAffinity::Cpu(core::marker::PhantomData)
                }
                #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
                PcuCallArgumentKind::ResidentRead(resident) => {
                    BorrowedAffinity::Device(resident.session)
                }
                #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
                PcuCallArgumentKind::ResidentWrite(resident) => {
                    BorrowedAffinity::Device(resident.session)
                }
                #[cfg(feature = "mlx")]
                PcuCallArgumentKind::MlxRead(resident) => BorrowedAffinity::Mlx(resident.root),
                #[cfg(feature = "mlx")]
                PcuCallArgumentKind::MlxWrite(resident) => BorrowedAffinity::Mlx(resident.root),
            };
            if affinity.is_some_and(|selected| !same(selected, root)) {
                return Err(super::PcuExecutionError::Argument(
                    super::super::PcuArgumentError::SessionMismatch,
                ));
            }
            affinity = Some(root);
        }
        Ok(affinity)
    }
    #[cfg(not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    )))]
    {
        let _ = arguments;
        Ok(None)
    }
}

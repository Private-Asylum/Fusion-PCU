//! Cold provider inventory and activation. Prepared warm execution never enters this module.

#[cfg(any(feature = "rocm", feature = "cuda"))]
use std::rc::Rc;
#[rustfmt::skip]
use super::{
    rank_candidates,
    Candidate,
    Provider,
    Session,
    PcuBackendChoice,
    PcuExecutionError,
    PcuExecutionPolicy,
};
#[cfg(feature = "mlx")]
#[rustfmt::skip]
use super::{
    MlxDiscovery,
    MlxError,
    collect_mlx_candidates,
    open_mlx,
};
#[cfg(feature = "cuda")]
#[rustfmt::skip]
use super::{
    CudaDiscovery,
    CudaOwnedDispatchBackend,
    collect_cuda_candidates,
};
#[cfg(feature = "rocm")]
#[rustfmt::skip]
use super::{
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    collect_rocm_candidates,
};
#[cfg(feature = "metal")]
#[rustfmt::skip]
use super::{
    MetalDiscovery,
    collect_metal_candidates,
    open_metal,
};
#[cfg(feature = "vulkan")]
#[rustfmt::skip]
use super::{
    PcuVulkanDiscovery,
    PcuVulkanError,
    collect_vulkan_candidates,
    open_vulkan,
};

#[cfg(feature = "cpu")]
#[rustfmt::skip]
use super::{
    collect_cpu_candidates,
    PcuCpuDiscovery,
    PcuCpuDiscoveryError,
};

pub(super) struct Discoveries {
    #[cfg(feature = "mlx")]
    mlx: Option<Result<MlxDiscovery, MlxError>>,
    #[cfg(feature = "cpu")]
    cpu: Option<Result<PcuCpuDiscovery, PcuCpuDiscoveryError>>,
    #[cfg(feature = "cuda")]
    cuda: Option<CudaDiscovery>,
    #[cfg(feature = "rocm")]
    rocm: Option<RocmDiscovery>,
    #[cfg(feature = "metal")]
    metal: Option<Result<MetalDiscovery, fusion_pcu_metal::MetalError>>,
    #[cfg(feature = "vulkan")]
    vulkan: Option<Result<PcuVulkanDiscovery, PcuVulkanError>>,
}

impl Discoveries {
    pub(super) fn new(policy: PcuExecutionPolicy) -> Self {
        #[cfg(not(any(
            feature = "metal",
            feature = "vulkan",
            feature = "cpu",
            feature = "mlx"
        )))]
        let _ = policy;
        Self {
            #[cfg(feature = "mlx")]
            mlx: matches!(
                policy.backend,
                PcuBackendChoice::Automatic | PcuBackendChoice::Mlx
            )
            .then(MlxDiscovery::discover_default),
            #[cfg(feature = "cpu")]
            cpu: matches!(
                policy.backend,
                PcuBackendChoice::Automatic | PcuBackendChoice::Cpu
            )
            .then(PcuCpuDiscovery::discover),
            #[cfg(feature = "cuda")]
            cuda: matches!(
                policy.backend,
                PcuBackendChoice::Automatic | PcuBackendChoice::Cuda
            )
            .then(CudaDiscovery::new),
            #[cfg(feature = "rocm")]
            rocm: matches!(
                policy.backend,
                PcuBackendChoice::Automatic | PcuBackendChoice::Rocm
            )
            .then(RocmDiscovery::new),
            #[cfg(feature = "metal")]
            metal: matches!(
                policy.backend,
                PcuBackendChoice::Automatic | PcuBackendChoice::Metal
            )
            .then(MetalDiscovery::discover),
            #[cfg(feature = "vulkan")]
            vulkan: matches!(
                policy.backend,
                PcuBackendChoice::Automatic | PcuBackendChoice::Vulkan
            )
            .then(PcuVulkanDiscovery::discover),
        }
    }

    pub(super) fn candidates(
        &self,
        policy: PcuExecutionPolicy,
        kernel: &crate::PcuDispatchKernelIr<'_>,
    ) -> (Vec<Candidate>, Vec<String>) {
        let mut candidates = Vec::new();
        let mut discovery_errors = Vec::new();

        #[cfg(feature = "mlx")]
        match &self.mlx {
            Some(Ok(discovery)) => {
                if let Err(error) =
                    collect_mlx_candidates(discovery, policy, kernel, &mut candidates)
                {
                    discovery_errors.push(format!("MLX discovery: {error}"));
                }
            }
            Some(Err(error)) => discovery_errors.push(format!("MLX discovery: {error}")),
            None => {}
        }

        #[cfg(feature = "cuda")]
        if let Some(discovery) = &self.cuda
            && let Err(error) =
                collect_cuda_candidates(discovery, policy, Some(kernel), &mut candidates)
        {
            discovery_errors.push(format!("CUDA discovery: {error}"));
        }
        #[cfg(feature = "rocm")]
        if let Some(discovery) = &self.rocm
            && let Err(error) =
                collect_rocm_candidates(discovery, policy, Some(kernel), &mut candidates)
        {
            discovery_errors.push(format!("ROCm discovery: {error}"));
        }
        #[cfg(feature = "metal")]
        match &self.metal {
            Some(Ok(discovery)) => {
                if let Err(error) =
                    collect_metal_candidates(discovery, policy, Some(kernel), &mut candidates)
                {
                    discovery_errors.push(format!("Metal discovery: {error}"));
                }
            }
            Some(Err(error)) => discovery_errors.push(format!("Metal discovery: {error}")),
            None => {}
        }
        #[cfg(feature = "vulkan")]
        match &self.vulkan {
            Some(Ok(discovery)) => {
                if let Err(error) =
                    collect_vulkan_candidates(discovery, policy, kernel, &mut candidates)
                {
                    discovery_errors.push(format!("Vulkan discovery: {error}"));
                }
            }
            Some(Err(error)) => discovery_errors.push(format!("Vulkan discovery: {error}")),
            None => {}
        }
        #[cfg(feature = "cpu")]
        match &self.cpu {
            Some(Ok(discovery)) => {
                if let Err(error) =
                    collect_cpu_candidates(discovery, policy, kernel, &mut candidates)
                {
                    discovery_errors.push(format!("CPU discovery: {error:?}"));
                }
            }
            Some(Err(error)) => discovery_errors.push(format!("CPU discovery: {error:?}")),
            None => {}
        }
        rank_candidates(&mut candidates);

        (candidates, discovery_errors)
    }

    pub(super) fn open(
        &self,
        candidate: &Candidate,
        policy: PcuExecutionPolicy,
    ) -> Result<Session, PcuExecutionError> {
        #[cfg(not(any(feature = "rocm", feature = "cuda")))]
        let _ = policy;
        match candidate.provider {
            #[cfg(feature = "mlx")]
            Provider::Mlx => open_mlx(self.mlx.as_ref(), candidate.device),
            #[cfg(feature = "cpu")]
            Provider::Cpu => self
                .cpu
                .as_ref()
                .and_then(|snapshot| snapshot.as_ref().ok())
                .map_or_else(
                    || Err(PcuExecutionError::NoBackendEnabled),
                    |discovery| {
                        crate::PcuDeviceActivation::open_device(discovery, candidate.device)
                            .map(Session::Cpu)
                            .map_err(PcuExecutionError::CpuDiscovery)
                    },
                ),
            #[cfg(feature = "vulkan")]
            Provider::Vulkan => open_vulkan(self.vulkan.as_ref(), candidate.device),
            #[cfg(feature = "metal")]
            Provider::Metal => open_metal(self.metal.as_ref(), candidate.device),
            #[cfg(feature = "cuda")]
            Provider::Cuda => {
                let discovery = self
                    .cuda
                    .as_ref()
                    .ok_or(PcuExecutionError::NoBackendEnabled)?;
                CudaOwnedDispatchBackend::open(discovery, candidate.device, policy.block_size)
                    .map(|backend| Session::Cuda(Rc::new(backend)))
                    .map_err(|error| {
                        PcuExecutionError::BackendFailure(format!("CUDA initialization: {error}"))
                    })
            }
            #[cfg(feature = "rocm")]
            Provider::Rocm => {
                let discovery = self
                    .rocm
                    .as_ref()
                    .ok_or(PcuExecutionError::NoBackendEnabled)?;
                RocmOwnedDispatchBackend::open(discovery, candidate.device, policy.block_size)
                    .map(|backend| Session::Rocm(Rc::new(backend)))
                    .map_err(|error| {
                        PcuExecutionError::BackendFailure(format!("ROCm initialization: {error}"))
                    })
            }
        }
    }
}

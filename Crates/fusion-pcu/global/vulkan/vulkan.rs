//! Cold Vulkan shader source and artifact preferences for ordinary `#[pcu]` calls.
//!
//! These process preferences apply to newly opened invocation sessions. Existing
//! resident owners keep their original device session and shader configuration;
//! changing preferences never retires their live pipelines. Cache policy does
//! not select a backend, change numerical permissions or enable CPU fallback.

#[rustfmt::skip]
pub use fusion_pcu_vulkan::{
    PcuVulkanShaderCachePolicy,
    PcuVulkanShaderDiskConfig,
    PcuVulkanShaderSource,
};

/// Source packaging and generated module retention are independent choices.
///
/// External composed packages can omit their embedded templates at build time.
/// Disk caching persists generated `.spv` files; it alone does not shrink a
/// binary or avoid the current exact-module lowering check on a cold hit.
#[derive(Clone, Debug, Default)]
pub struct PcuVulkanShaderOptions {
    pub source: PcuVulkanShaderSource,
    pub cache: PcuVulkanShaderCachePolicy,
}

impl PcuVulkanShaderOptions {
    pub(crate) fn apply(&self, backend: &fusion_pcu_vulkan::PcuVulkanBackend) {
        backend.configure_shader_source(self.source.clone());
        backend.configure_shader_cache(self.cache.clone());
    }
}

/// Replace shader preferences for later ordinary invocation calls.
///
/// Prepared-call selections invalidate through the existing policy generation.
/// No extra lock, file lookup or cache probe enters a retained warm call. The
/// source directory and disk permissions are checked during cold preparation,
/// so configuration itself performs no filesystem IO. Explicit prepared
/// backend instances remain independently configurable through their leaf API.
///
/// # Errors
/// Returns a poisoned-policy-lock or exhausted-generation error.
pub fn configure(options: PcuVulkanShaderOptions) -> Result<(), super::PcuExecutionError> {
    super::policy::configure_vulkan_shaders(options)
}

/// Restore embedded source and RAM-only generated-artifact caching.
///
/// External-only builds must select an external composed source before using
/// composed arithmetic; restoring this default does not invent missing assets.
///
/// # Errors
/// Returns the same configuration errors as [`configure`].
pub fn use_defaults() -> Result<(), super::PcuExecutionError> {
    configure(PcuVulkanShaderOptions::default())
}

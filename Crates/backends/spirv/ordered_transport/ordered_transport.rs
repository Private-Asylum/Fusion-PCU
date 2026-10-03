//! Separate cold static raw-carrier transport; no checked arithmetic or execution grant.
#[path = "compile/compile.rs"]
mod compile;
#[path = "emit/emit.rs"]
mod emit;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuImplementationRequirements,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    PcuSpirvComposedResource,
    PcuSpirvError,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvSink,
};
const RESOURCES: usize = 4;
const STEPS: usize = 64;
const SOURCE_REGISTERS: usize = 256;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Step {
    function: usize,
    arguments: [u32; 3],
}
impl Step {
    const EMPTY: Self = Self {
        function: 0,
        arguments: [0; 3],
    };
}
/// Provider-private static profile for 22 raw carriers, up to four actual resources and 64 steps.
///
/// Unused declarations do not allocate descriptors. Source SSA IDs are below 256; loads own their
/// value bits before later stores. Cross-lane mutable dependencies and Portable are refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcuSpirvOrderedTransportProfile {
    requirements: PcuImplementationRequirements,
    scalar: PcuScalarType,
    extent: u32,
    resources: [PcuSpirvComposedResource; RESOURCES],
    resource_count: usize,
    steps: [Step; STEPS],
    step_count: usize,
}
impl PcuSpirvOrderedTransportProfile {
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
    #[must_use]
    pub const fn scalar(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn extent(&self) -> u32 {
        self.extent
    }
    #[must_use]
    pub const fn element_bytes(&self) -> usize {
        (self.scalar.bit_width() / 8) as usize
    }
    #[must_use]
    pub const fn resources(&self) -> &[PcuSpirvComposedResource] {
        self.resources.split_at(self.resource_count).0
    }
    #[must_use]
    pub const fn step_count(&self) -> usize {
        self.step_count
    }
    /// One native invocation owns each complete output U32 word, including private padding.
    #[must_use]
    pub fn dispatch_extent(&self) -> u32 {
        (self.extent * u32::from(self.scalar.bit_width() / 8)).div_ceil(4)
    }
    #[must_use]
    pub fn local_id(&self) -> Option<u32> {
        PcuScalarType::ALL
            .iter()
            .position(|s| *s == self.scalar)
            .and_then(|ordinal| u32::try_from(ordinal).ok())
            .map(|ordinal| 19456 + ordinal)
    }
}
/// Detaches pure representation transport into cold native load/store call targets.
/// # Errors
/// Refuses malformed typing/roles/geometry, noncarriers, Portable and private capacity excess.
pub fn validate_ordered_scalar_transport_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvOrderedTransportProfile, PcuSpirvError> {
    compile::prepare(kernel)
}
/// Emits a statically frozen U32-only program; unused call slots are physically absent.
/// # Errors
/// Returns admission/version/capability/sink failures without granting a runtime profile.
pub fn lower_ordered_scalar_transport_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvOrderedTransportProfile), PcuSpirvError> {
    let profile = compile::prepare(kernel)?;
    let info = emit::lower(&profile, options, sink)?;
    Ok((info, profile))
}

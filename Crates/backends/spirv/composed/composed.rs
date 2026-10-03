//! Cold static native-function lowering for a bounded composed checked-float map.
//!
//! This compiler surface does not add Vulkan execution admission. Every native call
//! target and argument is frozen into ordinary SPIR-V constants before emission;
//! no instruction buffer or operation interpreter is traversed by the shader.
#[path = "compile/compile.rs"]
mod compile;
#[path = "emit/emit.rs"]
mod emit;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuCheckedScalarFaultLaw,
    PcuDispatchKernelIr,
    PcuImplementationRequirements,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    PcuSpirvError,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvSink,
};

const BINDINGS: usize = 4;
const STEPS: usize = 64;
const SOURCE_REGISTERS: usize = 256;

/// Actual unique resource in first-access order, independent of declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuSpirvComposedResource {
    pub binding: PcuBindingRef,
    pub declaration: usize,
    pub access: PcuBindingAccess,
    /// Required original view span; this profile conservatively seeds every read.
    pub read_elements: u32,
    pub write_elements: u32,
}
impl PcuSpirvComposedResource {
    const EMPTY: Self = Self {
        binding: PcuBindingRef::new(0, 0),
        declaration: 0,
        access: PcuBindingAccess::ReadOnly,
        read_elements: 0,
        write_elements: 0,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Step {
    function: usize,
    arguments: [u32; 8],
    law: Option<PcuCheckedScalarFaultLaw>,
}
impl Step {
    const EMPTY: Self = Self {
        function: 0,
        arguments: [0; 8],
        law: None,
    };
}

/// Detached cold compiler profile, with provider-private capacity bounds.
///
/// At most four declarations and 64 steps, source SSA IDs below 256, homogeneous
/// six-format scalar values are supported. The multi-effect constructor requires
/// at least two checked steps; the separate one-effect constructor requires exactly one.
/// Cross-index read/write hazards and Portable composition remain unadmitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcuSpirvComposedFloatProfile {
    requirements: PcuImplementationRequirements,
    scalar: PcuScalarType,
    extent: u32,
    declarations: [PcuBindingRef; BINDINGS],
    declaration_access: [PcuBindingAccess; BINDINGS],
    declaration_count: usize,
    resources: [PcuSpirvComposedResource; BINDINGS],
    resource_count: usize,
    steps: [Step; STEPS],
    step_count: usize,
    one_effect: bool,
}
impl PcuSpirvComposedFloatProfile {
    /// Separately admitted single checked effect; the multi-effect profile stays unchanged.
    #[must_use]
    pub const fn is_one_effect(&self) -> bool {
        self.one_effect
    }
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
    pub const fn declarations(&self) -> &[PcuBindingRef] {
        self.declarations.split_at(self.declaration_count).0
    }

    /// Original access also survives for declarations that the program never reads.
    #[must_use]
    pub fn declaration_access(&self, index: usize) -> Option<PcuBindingAccess> {
        self.declaration_access
            .get(index)
            .copied()
            .filter(|_| index < self.declaration_count)
    }

    #[must_use]
    pub const fn element_bytes(&self) -> usize {
        match self.scalar {
            PcuScalarType::F16 | PcuScalarType::BF16 => 2,
            PcuScalarType::F8E4M3FN | PcuScalarType::F8E5M2 => 1,
            PcuScalarType::F64 => 8,
            _ => 4,
        }
    }

    #[must_use]
    pub const fn resources(&self) -> &[PcuSpirvComposedResource] {
        self.resources.split_at(self.resource_count).0
    }

    /// Ordered instruction-local law; non-arithmetic steps cannot report a fault.
    #[must_use]
    pub fn fault_law(&self, step: usize) -> Option<PcuCheckedScalarFaultLaw> {
        self.steps.get(step).filter(|_| step < self.step_count)?.law
    }

    #[must_use]
    pub const fn step_count(&self) -> usize {
        self.step_count
    }

    #[must_use]
    pub const fn packing_lanes(&self) -> u32 {
        match self.scalar {
            PcuScalarType::F16 | PcuScalarType::BF16 => 2,
            PcuScalarType::F8E4M3FN | PcuScalarType::F8E5M2 => 4,
            _ => 1,
        }
    }

    #[must_use]
    pub const fn dispatch_extent(&self) -> u32 {
        self.extent.div_ceil(self.packing_lanes())
    }
}

/// Assesses exact structure and detaches static calls without allocating.
///
/// # Errors
/// Rejects unsupported numerical profiles, scalar types, malformed typed resources,
/// provider-private capacity excess and cross-index read/write dependencies.
pub fn validate_composed_float_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvComposedFloatProfile, PcuSpirvError> {
    compile::prepare(kernel, false)
}

/// Assesses exactly one observable checked operation, including an unused result.
///
/// # Errors
/// Rejects zero or multiple checked operations and the same unsupported structural,
/// resource and numerical requirements as the multi-effect compiler.
pub fn validate_one_effect_float_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvComposedFloatProfile, PcuSpirvError> {
    compile::prepare(kernel, true)
}

/// Emits a frozen native-function program using U32 arithmetic only.
///
/// # Errors
/// Returns structural/capability rejection before writing words, or a sink error.
pub fn lower_composed_float_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvComposedFloatProfile), PcuSpirvError> {
    let profile = compile::prepare(kernel, false)?;
    let module = emit::lower(&profile, options, sink)?;
    Ok((module, profile))
}

/// Emits a separately assessed one-effect static program using the same native helpers.
///
/// # Errors
/// Returns exact one-effect structural/capability rejection or a sink error.
pub fn lower_one_effect_float_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvComposedFloatProfile), PcuSpirvError> {
    let profile = compile::prepare(kernel, true)?;
    let module = emit::lower(&profile, options, sink)?;
    Ok((module, profile))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

//! Cold static native-function lowering for a bounded composed checked scalar map.
//!
//! This compiler surface does not add Vulkan execution admission. Every native call
//! target and argument is frozen into ordinary SPIR-V constants before emission;
//! no instruction buffer or operation interpreter is traversed by the shader.
#[path = "template/template.rs"]
mod template;
#[rustfmt::skip]
pub use template::{
    PcuSpirvComposedTemplateFamily,
    PcuSpirvComposedTemplateData,
    PcuSpirvComposedTemplate,
    PCU_COMPOSED_TEMPLATE_REVISION,
};
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
/// six-format float or fourteen integer scalar values are supported. The multi-effect constructor requires
/// at least two checked steps; the separate one-effect constructor requires exactly one.
/// Cross-index read/write hazards remain unadmitted; Portable composition is limited to the checked integer profiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcuSpirvComposedProfile {
    requirements: PcuImplementationRequirements,
    scalar: PcuScalarType,
    element_bytes: usize,
    packing_lanes: u32,
    extent: u32,
    declarations: [PcuBindingRef; BINDINGS],
    declaration_access: [PcuBindingAccess; BINDINGS],
    argument_access: [PcuBindingAccess; BINDINGS],
    declaration_count: usize,
    resources: [PcuSpirvComposedResource; BINDINGS],
    resource_count: usize,
    steps: [Step; STEPS],
    step_count: usize,
    one_effect: bool,
}
impl PcuSpirvComposedProfile {
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

    /// Caller access frozen cold; host mutable slices realize admitted integer write-only declarations.
    #[must_use]
    pub fn argument_access(&self, index: usize) -> Option<PcuBindingAccess> {
        self.argument_access
            .get(index)
            .copied()
            .filter(|_| index < self.declaration_count)
    }

    #[must_use]
    pub const fn element_bytes(&self) -> usize {
        self.element_bytes
    }

    #[must_use]
    pub const fn is_integer(&self) -> bool {
        matches!(
            self.scalar,
            PcuScalarType::U8
                | PcuScalarType::I8
                | PcuScalarType::U16
                | PcuScalarType::I16
                | PcuScalarType::U32
                | PcuScalarType::I32
                | PcuScalarType::U64
                | PcuScalarType::I64
                | PcuScalarType::U128
                | PcuScalarType::I128
                | PcuScalarType::U256
                | PcuScalarType::I256
                | PcuScalarType::U512
                | PcuScalarType::I512
        )
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
        self.packing_lanes
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
) -> Result<PcuSpirvComposedProfile, PcuSpirvError> {
    compile::prepare(kernel, false)
}

/// Assesses exactly one observable checked operation, including an unused result.
///
/// # Errors
/// Rejects zero or multiple checked operations and the same unsupported structural,
/// resource and numerical requirements as the multi-effect compiler.
pub fn validate_one_effect_float_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvComposedProfile, PcuSpirvError> {
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
) -> Result<(PcuSpirvModuleInfo, PcuSpirvComposedProfile), PcuSpirvError> {
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
) -> Result<(PcuSpirvModuleInfo, PcuSpirvComposedProfile), PcuSpirvError> {
    let profile = compile::prepare(kernel, true)?;
    let module = emit::lower(&profile, options, sink)?;
    Ok((module, profile))
}

/// Returns the built-in template for external package export.
#[cfg(feature = "embedded-composed")]
#[must_use]
pub const fn embedded_composed_template(
    family: PcuSpirvComposedTemplateFamily,
) -> PcuSpirvComposedTemplateData<'static> {
    emit::embedded_data(family)
}

/// Lowers from an exact external compiler template, without filesystem or hot source selection.
/// # Errors
/// Returns compiler-profile/capability or mismatched template-family refusal.
pub fn lower_composed_float_with_template<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    template: PcuSpirvComposedTemplate<'_>,
    one_effect: bool,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvComposedProfile), PcuSpirvError> {
    let profile = compile::prepare(kernel, one_effect)?;
    let module = emit::lower_external(&profile, options, sink, template)?;
    Ok((module, profile))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

/// Detaches a bounded homogeneous checked Add/Sub/Mul composition.
/// # Errors
/// Refuses unsupported scalars, malformed typed SSA/resources, mismatched ranges,
/// cross-index read/write, other operations and unqualified reproducibility.
pub fn validate_composed_integer_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvComposedProfile, PcuSpirvError> {
    compile::prepare_integer(kernel, false)
}
/// Emits the exact integer static-call template for an admitted composition.
/// # Errors
/// Returns validation, template, target or sink errors without widening admission.
pub fn lower_composed_integer_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvComposedProfile), PcuSpirvError> {
    let profile = validate_composed_integer_map(kernel)?;
    let info = emit::lower(&profile, options, sink)?;
    Ok((info, profile))
}
/// Emits integer composition with a sealed exact external helper asset.
/// # Errors
/// Returns compiler, family, target or sink errors; arbitrary shaders are refused.
pub fn lower_composed_integer_with_template<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    template: PcuSpirvComposedTemplate<'_>,
    one_effect: bool,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvComposedProfile), PcuSpirvError> {
    let profile = compile::prepare_integer(kernel, one_effect)?;
    let info = emit::lower_external(&profile, options, sink, template)?;
    Ok((info, profile))
}

/// Detaches one checked integer effect with ordered transport, including discarded results.
/// # Errors
/// Refuses unqualified wide one-effect Sub/Mul and invalid typed resource/numeric contracts.
pub fn validate_one_effect_integer_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvComposedProfile, PcuSpirvError> {
    compile::prepare_integer(kernel, true)
}
/// Emits an admitted one-effect integer map without dropping observable faults.
/// # Errors
/// Returns compiler/template/target/sink failures.
pub fn lower_one_effect_integer_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvComposedProfile), PcuSpirvError> {
    let profile = validate_one_effect_integer_map(kernel)?;
    let info = emit::lower(&profile, options, sink)?;
    Ok((info, profile))
}

//! Cold ordered checked composition; this plan alone grants no native execution.
#[rustfmt::skip]
use fusion_pcu::{
    assess_checked_float_map_resources,
    assess_checked_integer_map_resources,
    CheckedScalarMapResource,
    CheckedScalarMapResourceSchema,
    PcuBindingAccess,
    PcuBindingRef,
    PcuCheckedScalarFaultLaw,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuExecutionFault,
    PcuImplementationRequirements,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
use crate::MetalError;

/// An observable checked instruction, including one whose value is later discarded.
#[derive(Clone, Copy, Debug)]
pub struct MetalCheckedMapEffect {
    /// Position in the original flat instruction body; never a reordered SSA ordinal.
    pub instruction: usize,
    /// The exact local operation/range/underflow fault contract retained cold.
    pub law: PcuCheckedScalarFaultLaw,
}

/// Bounded detached metadata for future native checked helper composition.
///
/// Ten standard integer widths and F32/F64 are the initial candidate types. This
/// is pure cold assessment, not an executable offer, shader or discovery claim.
/// Four actual resources, two writers and 32 checked effects bound the candidate.
/// Full original declaration permissions and each checked instruction remain
/// intact; unaccessed declarations have no physical resource. A node with an
/// overwritten/unused result remains an observable effect with its own law.
#[derive(Clone, Debug)]
pub struct MetalCheckedMapPlan {
    schema: Box<CheckedScalarMapResourceSchema<4>>,
    declarations: Vec<(PcuBindingRef, PcuBindingAccess)>,
    instructions: Vec<PcuDispatchDataOp>,
    effects: Vec<MetalCheckedMapEffect>,
    status_bytes: usize,
}
impl MetalCheckedMapPlan {
    /// Validate exact resource, typed SSA, bounded geometry and local arithmetic policies.
    ///
    /// Portable and cross-index read/write dependencies are deliberately outside
    /// this candidate. An initial snapshot is distinct from required full view
    /// capacity; omitting original uploads requires a later native proof.
    /// # Errors
    /// Rejects unsupported dtype/request, invalid IR, resource/effect bounds,
    /// cross-index dependencies and unrepresentable payload/status extents.
    pub fn assess(
        kernel: &PcuDispatchKernelIr<'_>,
        scalar: PcuScalarType,
    ) -> Result<Self, MetalError> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        if kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            != PcuReproducibility::Unspecified
        {
            return Err(unsupported());
        }
        let value_type = PcuValueType::Scalar(scalar);
        let caps = PcuValueTypeCaps::for_scalar(scalar);
        let schema = match scalar {
            PcuScalarType::F32 | PcuScalarType::F64 => {
                assess_checked_float_map_resources::<4>(kernel, value_type, caps)
                    .map_err(|_| unsupported())?
            }
            PcuScalarType::I8
            | PcuScalarType::U8
            | PcuScalarType::I16
            | PcuScalarType::U16
            | PcuScalarType::I32
            | PcuScalarType::U32
            | PcuScalarType::I64
            | PcuScalarType::U64
            | PcuScalarType::I128
            | PcuScalarType::U128 => {
                assess_checked_integer_map_resources::<4>(kernel, value_type, caps)
                    .map_err(|_| unsupported())?
            }
            _ => return Err(unsupported()),
        };
        let resources = schema.resources();
        let writers = resources
            .iter()
            .filter(|role| role.minimum_write_elements != 0)
            .count();
        if writers == 0
            || writers > 2
            || resources
                .iter()
                .any(|role| role.has_cross_index_read_write())
        {
            return Err(unsupported());
        }
        let width = usize::from(scalar.bit_width()) / 8;
        for role in resources {
            validate_bytes(
                usize::try_from(role.minimum_elements()).map_err(|_| invalid_extent())?,
                width,
            )?;
        }
        let body = match kernel.ops {
            [PcuDispatchOp::GridStrideLoop { body, .. }, _] => *body,
            ops => &ops[..ops.len() - 1],
        };
        if body.len() > 128 {
            return Err(unsupported());
        }
        let (instructions, effects) = detach(body, scalar)?;
        let status_bytes = validate_bytes(
            usize::try_from(schema.logical_extent)
                .map_err(|_| invalid_extent())?
                .checked_mul(effects.len())
                .ok_or_else(invalid_extent)?,
            4,
        )?;
        Ok(Self {
            schema: Box::new(schema),
            declarations: kernel
                .bindings
                .iter()
                .map(|binding| (binding.reference(), binding.access))
                .collect(),
            instructions,
            effects,
            status_bytes,
        })
    }
    /// Homogeneous typed value domain retained from the exact validated request.
    #[must_use]
    pub const fn value_type(&self) -> PcuValueType {
        self.schema.value_type
    }
    /// Original launch count; grid-stride visited extent is separately retained.
    #[must_use]
    pub const fn submitted_invocations(&self) -> u32 {
        self.schema.submitted_invocations
    }
    /// Exact original function header, not inferred from global execution policy.
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.schema.requirements
    }
    #[must_use]
    pub const fn logical_extent(&self) -> u32 {
        self.schema.logical_extent
    }
    /// Includes untouched declarations with their original access metadata.
    #[must_use]
    pub const fn declared_bindings(&self) -> &[(PcuBindingRef, PcuBindingAccess)] {
        self.declarations.as_slice()
    }
    /// Actual unique resources, including incoming-content and full-view facts.
    #[must_use]
    pub fn resources(&self) -> &[CheckedScalarMapResource] {
        self.schema.resources()
    }
    /// Every ordered load/store/checked operation; no dead arithmetic elimination.
    #[must_use]
    pub const fn instructions(&self) -> &[PcuDispatchDataOp] {
        self.instructions.as_slice()
    }
    /// Independent per-instruction status laws in original effect order.
    #[must_use]
    pub const fn checked_effects(&self) -> &[MetalCheckedMapEffect] {
        self.effects.as_slice()
    }
    /// One `UInt32` diagnostic per logical lane per checked instruction.
    #[must_use]
    pub const fn status_byte_len(&self) -> usize {
        self.status_bytes
    }
    /// Emit a bounded original-header native candidate with a status column per effect.
    /// This cold source string does not itself prove execution or advertise an offer.
    /// # Errors
    /// Rejects instructions outside the initial binary native candidate.
    pub fn native_source(&self) -> Result<String, MetalError> {
        super::shader::lower(self)
    }
    /// Validate the future candidate's normalized step-major status bank before arbitration.
    ///
    /// Each checked instruction owns `logical_extent` `UInt32` words. The physical
    /// protocol uses zero success, overflow1/underflow3 with an optional recovered
    /// bit0x100, division2 and invalid-floating4. Every nonzero word must satisfy
    /// that exact instruction's law, including later words after an earlier fatal.
    /// This pure decoder is not proof a kernel wrote the bank or reached terminal
    /// completion. Those obligations remain with a separately qualified executor.
    /// # Errors
    /// Refuses wrong physical count, unknown encoding or an impossible local fault.
    pub fn validate_status_words(
        &self,
        words: &[u32],
    ) -> Result<Option<PcuExecutionFault>, MetalError> {
        if words.len().checked_mul(4) != Some(self.status_bytes) {
            return Err(unsupported());
        }
        let count = usize::try_from(self.logical_extent()).map_err(|_| invalid_extent())?;
        let mut fatal: Option<PcuExecutionFault> = None;
        let mut notice: Option<PcuExecutionFault> = None;
        for (effect, lane_words) in words.chunks_exact(count).enumerate() {
            for (lane, &record) in lane_words.iter().enumerate() {
                let kind = match record {
                    0 => continue,
                    1 | 0x101 => fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow,
                    2 => fusion_pcu::PcuExecutionFaultKind::DivideByZero,
                    3 | 0x103 => fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow,
                    4 => fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand,
                    _ => return Err(unsupported()),
                };
                let fault = PcuExecutionFault {
                    kind,
                    invocation_id: u64::try_from(lane).map_err(|_| invalid_extent())?,
                    recovered: record & 0x100 != 0,
                };
                if !self.accepts_record(effect, fault) {
                    return Err(unsupported());
                }
                let selected = if fault.recovered {
                    &mut notice
                } else {
                    &mut fatal
                };
                if selected.is_none_or(|prior| fault.invocation_id < prior.invocation_id) {
                    *selected = Some(fault);
                }
            }
        }
        Ok(fatal.or(notice))
    }
    /// Validate one decoded nonzero record before any cross-instruction arbitration.
    /// Raw encoding and terminal/native sibling publication remain backend obligations.
    #[must_use]
    pub fn accepts_record(&self, effect: usize, fault: PcuExecutionFault) -> bool {
        self.effects
            .get(effect)
            .is_some_and(|step| step.law.accepts(fault, u64::from(self.logical_extent())))
    }
}
fn validate_bytes(elements: usize, width: usize) -> Result<usize, MetalError> {
    let bytes = elements.checked_mul(width).ok_or_else(invalid_extent)?;
    if bytes == 0 || u32::try_from(bytes).is_err() {
        return Err(invalid_extent());
    }
    Ok(bytes)
}
const fn unsupported() -> MetalError {
    MetalError::Unsupported
}
const fn invalid_extent() -> MetalError {
    MetalError::InvalidExtent
}

fn detach(
    body: &[PcuDispatchOp<'_>],
    scalar: PcuScalarType,
) -> Result<(Vec<PcuDispatchDataOp>, Vec<MetalCheckedMapEffect>), MetalError> {
    let mut instructions = Vec::with_capacity(body.len());
    let mut effects = Vec::new();
    for (instruction, op) in body.iter().enumerate() {
        let PcuDispatchOp::Data(data) = *op else {
            return Err(unsupported());
        };
        let law = match data {
            PcuDispatchDataOp::CheckedIntegerBinary {
                op, range_policy, ..
            } => Some(
                PcuCheckedScalarFaultLaw::integer_binary(scalar, op, range_policy)
                    .ok_or_else(unsupported)?,
            ),
            PcuDispatchDataOp::CheckedFloatBinary {
                op,
                range_policy,
                underflow_policy,
                ..
            } => Some(
                PcuCheckedScalarFaultLaw::float_binary(scalar, op, range_policy, underflow_policy)
                    .ok_or_else(unsupported)?,
            ),
            PcuDispatchDataOp::CheckedFloatUnary {
                op,
                range_policy,
                underflow_policy,
                ..
            } => Some(
                PcuCheckedScalarFaultLaw::float_unary(scalar, op, range_policy, underflow_policy)
                    .ok_or_else(unsupported)?,
            ),
            PcuDispatchDataOp::BindingLoad { .. }
            | PcuDispatchDataOp::BindingStore { .. }
            | PcuDispatchDataOp::Constant { .. } => None,
            _ => return Err(unsupported()),
        };
        if let Some(law) = law {
            effects.push(MetalCheckedMapEffect { instruction, law });
        }
        instructions.push(data);
    }
    if effects.is_empty() || effects.len() > 32 {
        return Err(unsupported());
    }
    Ok((instructions, effects))
}

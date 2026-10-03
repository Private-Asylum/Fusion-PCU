//! Bounded cold byte-SSA transport admission, separate from arithmetic maps.

use std::fmt::Write;
#[rustfmt::skip]
use fusion_pcu::{
    describe_scalar_transport_map,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuImplementationRequirements,
    PcuReproducibility,
    PcuScalarTransportDescription,
    PcuScalarTransportResource,
    PcuScalarType,
    PcuValueType,
};
use crate::MetalError;

/// Exact ordered scalar transport through at most four actual storage resources.
///
/// The plan retains all original declared access/type roles and the complete request.
/// Unused declarations have no native storage slot. All 22 byte-addressed sealed
/// carriers copy their complete representation, including nonfinite floating bits.
/// Portable requests and cross-index read/write hazards remain unqualified.
#[derive(Debug, Clone)]
pub struct MetalTransportPlan {
    pub(super) description: PcuScalarTransportDescription<4>,
    pub(super) declarations: Vec<(PcuBindingRef, PcuBindingAccess)>,
    pub(super) shader: String,
    pub(super) width: usize,
}

impl MetalTransportPlan {
    pub(crate) fn assess_kernel(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MetalError> {
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|binding| binding.binding_type)
        else {
            return Err(MetalError::Unsupported);
        };
        Self::assess(kernel, scalar)
    }
    /// Complete original declarations; unused entries retain their typed access law.
    #[must_use]
    pub const fn declared_bindings(&self) -> &[(PcuBindingRef, PcuBindingAccess)] {
        self.declarations.as_slice()
    }
    /// Freezes exact actual roles, extents and typed SSA before any Metal discovery.
    /// # Errors
    /// Refuses packed/non-byte carriers, Portable, more than two actual writers,
    /// more than four resources or 128 instructions, cross-lane hazards or invalid IR.
    pub fn assess(
        kernel: &PcuDispatchKernelIr<'_>,
        scalar: PcuScalarType,
    ) -> Result<Self, MetalError> {
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) || kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            != PcuReproducibility::Unspecified
        {
            return Err(MetalError::Unsupported);
        }
        let description = describe_scalar_transport_map::<4>(kernel, scalar)
            .map_err(|_| MetalError::Unsupported)?;
        let resources = description.resources();
        if resources
            .iter()
            .any(|resource| resource.has_cross_index_read_write())
            || resources
                .iter()
                .filter(|resource| resource.minimum_write_elements != 0)
                .count()
                > 2
        {
            return Err(MetalError::Unsupported);
        }
        let width = usize::from(scalar.bit_width()) / 8;
        for resource in resources {
            let bytes = usize::try_from(resource.minimum_elements())
                .ok()
                .and_then(|elements| elements.checked_mul(width))
                .ok_or(MetalError::InvalidExtent)?;
            if bytes == 0 || u32::try_from(bytes).is_err() {
                return Err(MetalError::InvalidExtent);
            }
        }
        let body = match kernel.ops {
            [PcuDispatchOp::GridStrideLoop { body, .. }, _] => *body,
            ops => &ops[..ops.len() - 1],
        };
        if body.len() > 128 {
            return Err(MetalError::Unsupported);
        }
        Ok(Self {
            shader: shader(body, resources, width)?,
            description,
            declarations: kernel
                .bindings
                .iter()
                .map(|binding| (binding.reference(), binding.access))
                .collect(),
            width,
        })
    }

    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.description.scalar
    }

    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.description.requirements
    }

    #[must_use]
    pub const fn element_count(&self) -> u32 {
        self.description.logical_extent
    }

    /// Actual unique resources in original first-access order, including initial-read facts.
    #[must_use]
    pub fn resources(&self) -> &[PcuScalarTransportResource] {
        self.description.resources()
    }
}

fn shader(
    body: &[PcuDispatchOp<'_>],
    resources: &[PcuScalarTransportResource],
    width: usize,
) -> Result<String, MetalError> {
    let mut source = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void pcu_scalar_transport(\n",
    );
    for slot in 0..4 {
        writeln!(source, "device uchar* r{slot} [[buffer({slot})]],")
            .map_err(|_| MetalError::Unsupported)?;
    }
    source.push_str("constant uint2& config [[buffer(4)]], uint tid [[thread_position_in_grid]]) {\nif (tid >= config.x) return;\n");
    for instruction in body {
        let result = match *instruction {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                binding,
                index,
                result,
                ..
            }) => {
                let slot = resources
                    .iter()
                    .position(|resource| resource.binding == binding)
                    .ok_or(MetalError::Unsupported)?;
                let offset = if index == PcuDispatchIndex::BindingElementZero {
                    "0u"
                } else {
                    "tid"
                };
                // A private value owns the complete encoding now; a later source store
                // cannot change it, even when loading a mutable resource.
                writeln!(
                    source,
                    "uchar v{}[{width}]; for (uint b=0u;b<{width}u;++b) v{}[b]=r{slot}[{offset}*{width}u+b];",
                    result.0, result.0
                )
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
                let slot = resources
                    .iter()
                    .position(|resource| resource.binding == binding)
                    .ok_or(MetalError::Unsupported)?;
                writeln!(
                    source,
                    "for (uint b=0u;b<{width}u;++b) r{slot}[tid*{width}u+b]=v{}[b];",
                    value.0
                )
            }
            _ => return Err(MetalError::Unsupported),
        };
        result.map_err(|_| MetalError::Unsupported)?;
    }
    source.push_str("}\n");
    Ok(source)
}

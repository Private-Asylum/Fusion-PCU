//! Detached actual-binding roles for an already validated transport body.

use super::super::PcuScalarTransportError as Error;
use crate::map_validation::checked_scalar_resources;
#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuImplementationRequirements,
    PcuScalarType,
    PcuValueType,
    CheckedScalarMapResourceError,
};

/// One actual transport resource, preserving its declared Rust access role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuScalarTransportResource {
    pub binding: PcuBindingRef,
    pub declared_access: PcuBindingAccess,
    /// Required read view, regardless of whether prior stores supply contents.
    pub minimum_read_elements: u32,
    /// Incoming contents needed before a same-lane store supplies a later load.
    /// Cross-lane element-zero loads remain conservative for extent greater than one.
    pub minimum_initial_read_elements: u32,
    pub reads_element_zero: bool,
    pub minimum_write_elements: u32,
}

impl PcuScalarTransportResource {
    const EMPTY: Self = Self {
        binding: PcuBindingRef::new(0, 0),
        declared_access: PcuBindingAccess::ReadOnly,
        minimum_read_elements: 0,
        minimum_initial_read_elements: 0,
        reads_element_zero: false,
        minimum_write_elements: 0,
    };

    /// Required resource view capacity; byte size/alignment remain independent.
    #[must_use]
    pub const fn minimum_elements(self) -> u32 {
        if self.minimum_read_elements > self.minimum_write_elements {
            self.minimum_read_elements
        } else {
            self.minimum_write_elements
        }
    }

    /// Structural cross-index dependency requiring independent provider safety.
    /// This flag is neither a physical alias proof nor permission to race.
    #[must_use]
    pub const fn has_cross_index_read_write(self) -> bool {
        self.reads_element_zero && self.minimum_write_elements > 1
    }
}

/// First-access-ordered, detached cold facts; never an executable provider offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuScalarTransportDescription<const CAPACITY: usize> {
    resources: [PcuScalarTransportResource; CAPACITY],
    resource_count: usize,
    pub scalar: PcuScalarType,
    /// The complete unchanged numerical request, including unused permissions.
    pub requirements: PcuImplementationRequirements,
    pub submitted_invocations: u32,
    pub logical_extent: u32,
}

impl<const CAPACITY: usize> PcuScalarTransportDescription<CAPACITY> {
    /// Actual unique resources. Unused declarations consume no description slot.
    /// Original binding references must be retained when mapping any backend ABI.
    #[must_use]
    pub fn resources(&self) -> &[PcuScalarTransportResource] {
        &self.resources[..self.resource_count]
    }

    #[must_use]
    pub fn resource(&self, binding: PcuBindingRef) -> Option<PcuScalarTransportResource> {
        self.resources()
            .iter()
            .find(|resource| resource.binding == binding)
            .copied()
    }
}

pub(super) fn project<const CAPACITY: usize>(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
    body: &[PcuDispatchOp<'_>],
    logical_extent: u32,
) -> Result<PcuScalarTransportDescription<CAPACITY>, Error> {
    let schema = checked_scalar_resources::project_body::<CAPACITY, _>(
        kernel,
        PcuValueType::Scalar(scalar),
        body,
        logical_extent,
        Error::InvalidBinding,
    )
    .map_err(|error| match error {
        CheckedScalarMapResourceError::InvalidMap(error) => error,
        CheckedScalarMapResourceError::InsufficientCapacity {
            capacity,
            required_at_least,
        } => Error::InsufficientCapacity {
            capacity,
            required_at_least,
        },
    })?;
    let mut description = PcuScalarTransportDescription {
        resources: [PcuScalarTransportResource::EMPTY; CAPACITY],
        resource_count: schema.resources().len(),
        scalar,
        requirements: schema.requirements,
        submitted_invocations: schema.submitted_invocations,
        logical_extent: schema.logical_extent,
    };
    for (destination, resource) in description.resources.iter_mut().zip(schema.resources()) {
        *destination = PcuScalarTransportResource {
            binding: resource.binding,
            declared_access: resource.declared_access,
            minimum_read_elements: resource.minimum_read_elements,
            minimum_initial_read_elements: resource.minimum_initial_read_elements,
            reads_element_zero: resource.reads_element_zero,
            minimum_write_elements: resource.minimum_write_elements,
        };
    }
    Ok(description)
}

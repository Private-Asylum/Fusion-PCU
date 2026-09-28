//! Backend-neutral tensor storage requirements and overlap validation.

use alloc::vec::Vec;
#[rustfmt::skip]
use crate::{
    PcuMemoryAccess,
    PcuMemoryOverlap,
    PcuMemoryRange,
    PcuMemoryResource,
};

use super::ValueId;

/// Backend-neutral storage facts for graph values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorGraphRequirements<'a> {
    /// Dense row-major f32 values are the only layout and element type currently represented.
    pub values: Vec<TensorValueRequirement<'a>>,
    /// Sum of all graph-value output storage, not a peak/live memory estimate.
    pub total_value_bytes: Option<usize>,
}

/// Inclusive lifetime and output-storage facts for one value in an execution plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorValueLiveness {
    pub value: ValueId,
    pub output_bytes: Option<usize>,
    /// Index in `TensorExecutionPlan::node_order` where this value is produced.
    pub first_live_node: usize,
    /// Last node that reads this value, or its production node when it has no consumers.
    pub last_live_node: usize,
}

/// A pair of values that must occupy disjoint provider storage during the selected execution.
///
/// Read-only inputs/constants may share storage with each other. Every computed value is treated
/// as writable because the tensor dialect currently defines no in-place operation permissions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorStorageConstraint {
    pub left: ValueId,
    pub right: ValueId,
    pub left_bytes: u64,
    pub right_bytes: u64,
}

/// Why validation could not prove a requested storage relationship legal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorStorageValidationError {
    MissingResource(ValueId),
    ResourceTooSmall {
        value: ValueId,
        required_bytes: u64,
        available_bytes: u64,
    },
    InsufficientAccess {
        value: ValueId,
        required: PcuMemoryAccess,
        available: PcuMemoryAccess,
    },
    Overlapping {
        left: ValueId,
        right: ValueId,
    },
    UnknownOverlap {
        left: ValueId,
        right: ValueId,
    },
}

impl TensorStorageConstraint {
    /// Checks this requirement against provider-owned resources. Unknown overlap is rejected.
    ///
    /// # Errors
    ///
    /// Returns a missing-resource error when a value has no binding, or an overlap error unless
    /// the provider proves the two full ranges are disjoint.
    pub fn validate<R: PcuMemoryResource>(
        &self,
        resources: &[(ValueId, &R)],
    ) -> Result<(), TensorStorageValidationError> {
        let left = resources
            .iter()
            .find(|(value, _)| *value == self.left)
            .map(|(_, resource)| *resource)
            .ok_or(TensorStorageValidationError::MissingResource(self.left))?;
        let right = resources
            .iter()
            .find(|(value, _)| *value == self.right)
            .map(|(_, resource)| *resource)
            .ok_or(TensorStorageValidationError::MissingResource(self.right))?;
        self.validate_resources(left, right)
    }

    /// Checks this requirement when both resources have already been resolved by value ID.
    /// Unknown overlap is rejected conservatively.
    ///
    /// # Errors
    ///
    /// Returns an overlap error unless the provider proves the two full ranges are disjoint.
    pub fn validate_resources<R: PcuMemoryResource>(
        &self,
        left: &R,
        right: &R,
    ) -> Result<(), TensorStorageValidationError> {
        for (value, required_bytes, available_bytes) in [
            (self.left, self.left_bytes, left.size_bytes()),
            (self.right, self.right_bytes, right.size_bytes()),
        ] {
            if available_bytes < required_bytes {
                return Err(TensorStorageValidationError::ResourceTooSmall {
                    value,
                    required_bytes,
                    available_bytes,
                });
            }
        }
        let overlap = left.overlap(
            right,
            PcuMemoryRange {
                offset_bytes: 0,
                size_bytes: self.left_bytes,
            },
            PcuMemoryRange {
                offset_bytes: 0,
                size_bytes: self.right_bytes,
            },
        );
        match overlap {
            PcuMemoryOverlap::Disjoint => Ok(()),
            PcuMemoryOverlap::Overlapping => Err(TensorStorageValidationError::Overlapping {
                left: self.left,
                right: self.right,
            }),
            PcuMemoryOverlap::Unknown => Err(TensorStorageValidationError::UnknownOverlap {
                left: self.left,
                right: self.right,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorValueRequirement<'a> {
    pub value: ValueId,
    pub shape: &'a [usize],
    pub output_bytes: Option<usize>,
}

/// Access and dense-layout extent for one value's backing resource in a selected execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorValueStorageRequirement {
    pub value: ValueId,
    pub output_bytes: u64,
    pub access: PcuMemoryAccess,
}

/// One transient tensor value's assignment to reusable scratch storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorScratchStorageAssignment {
    pub value: ValueId,
    pub slot: usize,
}

/// Capacity and alignment required for one reusable scratch allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorScratchStorageSlot {
    pub capacity_bytes: usize,
    pub alignment_bytes: usize,
}

/// Cold storage plan for selected transient values in a serial, same-queue schedule.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorScratchStoragePlan {
    assignments: Vec<TensorScratchStorageAssignment>,
    slots: Vec<TensorScratchStorageSlot>,
    total_bytes: usize,
}

impl TensorScratchStoragePlan {
    /// Value-to-slot assignments in stable first-live order.
    #[must_use]
    pub fn assignments(&self) -> &[TensorScratchStorageAssignment] {
        &self.assignments
    }

    /// Scratch allocation requirements in stable slot order.
    #[must_use]
    pub fn slots(&self) -> &[TensorScratchStorageSlot] {
        &self.slots
    }

    /// Sum of slot capacities, checked while constructing this plan.
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Returns the scratch slot assigned to one selected value.
    #[must_use]
    pub fn slot_for(&self, value: ValueId) -> Option<usize> {
        self.assignments
            .iter()
            .find(|assignment| assignment.value == value)
            .map(|assignment| assignment.slot)
    }
}

const TENSOR_SCRATCH_ALIGNMENT_BYTES: usize = core::mem::size_of::<f32>();

/// Plans aligned reusable slots for eligible, non-escaping transient values.
///
/// `eligible` is supplied by the selected lowering adapter so it can omit values fused into
/// another operation or represented without storage. Inputs and requested outputs are filtered
/// by `TensorExecutionPlan` before this function is called. Lifetimes are inclusive: a prior
/// value is reusable only when it dies before the new value is produced. This deliberately
/// prohibits in-place aliasing when an operation reads one value and produces another.
pub(super) fn plan_scratch_storage(
    liveness: &[TensorValueLiveness],
    eligible: &[ValueId],
) -> Result<TensorScratchStoragePlan, super::TensorError> {
    let mut plan = TensorScratchStoragePlan::default();
    // Each slot's most recent lifetime is tracked separately from its public allocation facts.
    let mut slot_last_live = Vec::new();

    for life in liveness {
        if !eligible.contains(&life.value) {
            continue;
        }
        let bytes = life.output_bytes.ok_or(super::TensorError::ShapeOverflow)?;
        let aligned_bytes = bytes
            .checked_add(TENSOR_SCRATCH_ALIGNMENT_BYTES - 1)
            .map(|value| value & !(TENSOR_SCRATCH_ALIGNMENT_BYTES - 1))
            .ok_or(super::TensorError::ShapeOverflow)?;

        // Best fit avoids wasting large slots when a smaller one is sufficient.
        let slot = plan
            .slots
            .iter()
            .enumerate()
            .filter(|(index, slot)| {
                slot.capacity_bytes >= aligned_bytes
                    && slot_last_live[*index] < life.first_live_node
            })
            .min_by_key(|(index, slot)| (slot.capacity_bytes, *index))
            .map(|(index, _)| index);

        let slot = if let Some(slot) = slot {
            slot_last_live[slot] = life.last_live_node;
            slot
        } else if let Some((slot, previous_capacity)) = plan
            .slots
            .iter()
            .enumerate()
            .filter(|(index, slot)| {
                slot.capacity_bytes < aligned_bytes && slot_last_live[*index] < life.first_live_node
            })
            .max_by_key(|(index, slot)| (slot.capacity_bytes, core::cmp::Reverse(*index)))
            .map(|(index, slot)| (index, slot.capacity_bytes))
        {
            // Reuse the largest free undersized slot, minimizing the added capacity.
            plan.total_bytes = plan
                .total_bytes
                .checked_add(aligned_bytes - previous_capacity)
                .ok_or(super::TensorError::ShapeOverflow)?;
            plan.slots[slot].capacity_bytes = aligned_bytes;
            slot_last_live[slot] = life.last_live_node;
            slot
        } else {
            plan.total_bytes = plan
                .total_bytes
                .checked_add(aligned_bytes)
                .ok_or(super::TensorError::ShapeOverflow)?;
            plan.slots.push(TensorScratchStorageSlot {
                capacity_bytes: aligned_bytes,
                alignment_bytes: TENSOR_SCRATCH_ALIGNMENT_BYTES,
            });
            slot_last_live.push(life.last_live_node);
            plan.slots.len() - 1
        };

        plan.assignments.push(TensorScratchStorageAssignment {
            value: life.value,
            slot,
        });
    }
    Ok(plan)
}

impl TensorValueStorageRequirement {
    /// Checks a resource for dense storage capacity and the required access.
    ///
    /// Backends with compact physical layouts must adapt the extent before calling this method.
    ///
    /// # Errors
    ///
    /// Returns the resource's insufficient extent or access permission.
    pub fn validate<R: PcuMemoryResource>(
        &self,
        resource: &R,
    ) -> Result<(), TensorStorageValidationError> {
        let available_bytes = resource.size_bytes();
        if available_bytes < self.output_bytes {
            return Err(TensorStorageValidationError::ResourceTooSmall {
                value: self.value,
                required_bytes: self.output_bytes,
                available_bytes,
            });
        }
        let actual = resource.access();
        let permitted = match self.access {
            PcuMemoryAccess::ReadOnly => {
                matches!(
                    actual,
                    PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
                )
            }
            PcuMemoryAccess::WriteOnly => {
                matches!(
                    actual,
                    PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
                )
            }
            PcuMemoryAccess::ReadWrite => actual == PcuMemoryAccess::ReadWrite,
        };
        if permitted {
            Ok(())
        } else {
            Err(TensorStorageValidationError::InsufficientAccess {
                value: self.value,
                required: self.access,
                available: actual,
            })
        }
    }
}

pub fn node_output_bytes(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(core::mem::size_of::<f32>(), |bytes, &dimension| {
            bytes.checked_mul(dimension)
        })
}

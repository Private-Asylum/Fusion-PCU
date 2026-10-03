//! Backend-neutral tensor storage requirements and overlap validation.

use alloc::vec::Vec;
#[rustfmt::skip]
use crate::{
    core::PcuScalarType,
    PcuMemoryAccess,
    PcuMemoryOverlap,
    PcuMemoryRange,
    PcuMemoryResource,
};

use super::ValueId;

/// Why a selected graph cannot prove a consumed input may be overwritten by its `ReLU` result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorStorageReuseError {
    ValueNotInProgram(ValueId),
    ShapeOverflow(ValueId),
    UnsupportedScalarType {
        value: ValueId,
        scalar_type: PcuScalarType,
    },
    InputIsNotGraphInput(ValueId),
    OutputIsNotSelectedOutput(ValueId),
    InputIsSelectedOutput(ValueId),
    OutputHasSelectedConsumers {
        value: ValueId,
        actual: usize,
    },
    InputUseCount {
        value: ValueId,
        actual: usize,
    },
    DonorAndOtherAreSameValue(ValueId),
    NotTerminalBinary {
        donor: ValueId,
        other: ValueId,
        output: ValueId,
    },
    NotDirectRelu {
        input: ValueId,
        output: ValueId,
    },
    ShapeMismatch {
        input: ValueId,
        output: ValueId,
    },
    ScalarTypeMismatch {
        input: ValueId,
        output: ValueId,
    },
    LayoutMismatch {
        input: ValueId,
        output: ValueId,
    },
}

/// Binary operation whose terminal result may be considered for donor reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorBinaryOperation {
    Add,
    Sub,
    Mul,
}

/// Position of the consumed donor among a binary operation's ordered operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorBinaryOperand {
    Left,
    Right,
}

/// Core proof that a selected same-index `ReLU` may destructively reuse a consumed input.
///
/// This establishes graph-level legality only. It does not prove that a physical resource is
/// uniquely owned, quiescent, or usable as an in-place binding by a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorInputReuseProof {
    graph_id: u64,
    input: ValueId,
    output: ValueId,
    scalar_type: PcuScalarType,
    bytes: u64,
    alignment_bytes: u64,
}

impl TensorInputReuseProof {
    pub(super) const fn new(
        graph_id: u64,
        input: ValueId,
        output: ValueId,
        scalar_type: PcuScalarType,
        bytes: u64,
        alignment_bytes: u64,
    ) -> Self {
        Self {
            graph_id,
            input,
            output,
            scalar_type,
            bytes,
            alignment_bytes,
        }
    }

    /// Graph identity that anchors both value IDs in this proof.
    #[must_use]
    pub const fn graph_id(self) -> u64 {
        self.graph_id
    }

    /// Consumed external graph input whose bytes may be overwritten.
    #[must_use]
    pub const fn input(self) -> ValueId {
        self.input
    }

    /// Selected terminal `ReLU` output that may reuse the input bytes.
    #[must_use]
    pub const fn output(self) -> ValueId {
        self.output
    }

    /// Scalar representation shared by the input and output.
    #[must_use]
    pub const fn scalar_type(self) -> PcuScalarType {
        self.scalar_type
    }

    /// Exact dense byte extent proven for both values.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Dense alignment required by both values.
    #[must_use]
    pub const fn alignment_bytes(self) -> u64 {
        self.alignment_bytes
    }
}

/// Graph-level proof that one terminal binary result may overwrite its designated input.
///
/// This records the operation's operand order as well as dense storage facts. It does not prove
/// physical uniqueness, quiescence, or that a provider supports an in-place binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalBinaryDonorProof {
    graph_id: u64,
    donor: ValueId,
    other: ValueId,
    output: ValueId,
    operation: TensorBinaryOperation,
    donor_operand: TensorBinaryOperand,
    scalar_type: PcuScalarType,
    bytes: u64,
    alignment_bytes: u64,
}

impl TerminalBinaryDonorProof {
    #[allow(clippy::too_many_arguments)] // Captures the complete immutable proof facts once.
    pub(super) const fn new(
        graph_id: u64,
        donor: ValueId,
        other: ValueId,
        output: ValueId,
        operation: TensorBinaryOperation,
        donor_operand: TensorBinaryOperand,
        scalar_type: PcuScalarType,
        bytes: u64,
        alignment_bytes: u64,
    ) -> Self {
        Self {
            graph_id,
            donor,
            other,
            output,
            operation,
            donor_operand,
            scalar_type,
            bytes,
            alignment_bytes,
        }
    }

    /// Graph identity anchoring all values in this proof.
    #[must_use]
    pub const fn graph_id(self) -> u64 {
        self.graph_id
    }

    /// Consumed input whose storage may be overwritten.
    #[must_use]
    pub const fn donor(self) -> ValueId {
        self.donor
    }

    /// Other read-only operand.
    #[must_use]
    pub const fn other(self) -> ValueId {
        self.other
    }

    /// Selected terminal result.
    #[must_use]
    pub const fn output(self) -> ValueId {
        self.output
    }

    /// Operation represented by the proof.
    #[must_use]
    pub const fn operation(self) -> TensorBinaryOperation {
        self.operation
    }

    /// Position of the donor operand; required to preserve noncommutative operand order.
    #[must_use]
    pub const fn donor_operand(self) -> TensorBinaryOperand {
        self.donor_operand
    }

    /// Shared scalar representation.
    #[must_use]
    pub const fn scalar_type(self) -> PcuScalarType {
        self.scalar_type
    }

    /// Exact dense byte extent of donor and output.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Required dense alignment.
    #[must_use]
    pub const fn alignment_bytes(self) -> u64 {
        self.alignment_bytes
    }
}

/// Backend-neutral storage facts for graph values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorGraphRequirements<'a> {
    /// Dense row-major per-value scalar and layout facts. These describe graph/storage metadata;
    /// they do not claim that a reference or device executor supports every scalar operation.
    pub values: Vec<TensorValueRequirement<'a>>,
    /// Sum of all graph-value output storage, not a peak/live memory estimate.
    pub total_value_bytes: Option<usize>,
}

/// Inclusive lifetime and typed output-storage facts for one value in an execution plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorValueLiveness {
    pub value: ValueId,
    pub scalar_type: PcuScalarType,
    pub output_bytes: Option<usize>,
    pub alignment_bytes: Option<usize>,
    /// Index in `TensorExecutionPlan::node_order` where this value is produced.
    pub first_live_node: usize,
    /// Last node that reads this value, or its production node when it has no consumers.
    pub last_live_node: usize,
}

/// A pair of values that must occupy disjoint provider storage during the selected execution.
///
/// Read-only inputs/constants may share storage with each other. Computed values remain disjoint
/// under ordinary validation. A [`TensorInputReuseProof`] or [`TerminalBinaryDonorProof`] is
/// separate evidence for a narrow consuming case and does not relax these default constraints.
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
    InsufficientAlignment {
        value: ValueId,
        required_bytes: u64,
        available_bytes: u64,
    },
    InvalidAlignment {
        value: ValueId,
        required_bytes: u64,
        available_bytes: u64,
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
    pub scalar_type: PcuScalarType,
    pub output_bytes: Option<usize>,
    pub alignment_bytes: Option<usize>,
}

/// Typed dense-layout extent, alignment, and access for one selected value's backing resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorValueStorageRequirement {
    pub value: ValueId,
    pub scalar_type: PcuScalarType,
    pub output_bytes: u64,
    pub alignment_bytes: u64,
    pub access: PcuMemoryAccess,
}

/// One transient tensor value's assignment to reusable scratch storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorScratchStorageAssignment {
    pub value: ValueId,
    pub slot: usize,
}

/// Capacity and maximum alignment required for one reusable scratch allocation.
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
        let Some((_, layout_alignment)) = dense_scalar_layout(life.scalar_type) else {
            return Err(super::TensorError::UnsupportedScalarType {
                value: life.value,
                scalar_type: life.scalar_type,
            });
        };
        let bytes = life.output_bytes.ok_or(super::TensorError::ShapeOverflow)?;
        let alignment_bytes = life.alignment_bytes.unwrap_or(layout_alignment);
        if alignment_bytes < layout_alignment
            || alignment_bytes == 0
            || !alignment_bytes.is_power_of_two()
            || !alignment_bytes.is_multiple_of(layout_alignment)
        {
            return Err(super::TensorError::InvalidStorageAlignment {
                value: life.value,
                alignment_bytes,
            });
        }
        let aligned_bytes =
            align_up(bytes, alignment_bytes).ok_or(super::TensorError::ShapeOverflow)?;

        // Best fit avoids wasting large slots when a smaller one is sufficient.
        let slot = plan
            .slots
            .iter()
            .enumerate()
            .filter(|(index, slot)| {
                slot.capacity_bytes >= aligned_bytes
                    && slot.alignment_bytes >= alignment_bytes
                    && slot_last_live[*index] < life.first_live_node
            })
            .min_by_key(|(index, slot)| (slot.capacity_bytes, *index))
            .map(|(index, _)| index);

        let slot = if let Some(slot) = slot {
            slot_last_live[slot] = life.last_live_node;
            slot
        } else if let Some((slot, previous_capacity, previous_alignment)) = plan
            .slots
            .iter()
            .enumerate()
            .filter(|(index, slot)| {
                slot_last_live[*index] < life.first_live_node
                    && (slot.capacity_bytes < aligned_bytes
                        || slot.alignment_bytes < alignment_bytes)
            })
            .max_by_key(|(index, slot)| {
                (
                    slot.capacity_bytes.min(aligned_bytes),
                    slot.alignment_bytes.min(alignment_bytes),
                    core::cmp::Reverse(*index),
                )
            })
            .map(|(index, slot)| (index, slot.capacity_bytes, slot.alignment_bytes))
        {
            // Upgrade the most compatible free slot, accounting for capacity and alignment.
            // Upgrading the alignment is sound because the backing allocation is requested cold
            // from this final slot descriptor; no earlier assignment is submitted yet.
            let upgraded_alignment = previous_alignment.max(alignment_bytes);
            let upgraded_capacity = align_up(previous_capacity.max(bytes), upgraded_alignment)
                .ok_or(super::TensorError::ShapeOverflow)?;
            plan.total_bytes = plan
                .total_bytes
                .checked_add(upgraded_capacity - previous_capacity)
                .ok_or(super::TensorError::ShapeOverflow)?;
            plan.slots[slot].capacity_bytes = upgraded_capacity;
            plan.slots[slot].alignment_bytes = upgraded_alignment;
            slot_last_live[slot] = life.last_live_node;
            slot
        } else {
            plan.total_bytes = plan
                .total_bytes
                .checked_add(aligned_bytes)
                .ok_or(super::TensorError::ShapeOverflow)?;
            plan.slots.push(TensorScratchStorageSlot {
                capacity_bytes: aligned_bytes,
                alignment_bytes,
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
        let available_alignment = resource.alignment_bytes();
        if self.alignment_bytes == 0
            || !self.alignment_bytes.is_power_of_two()
            || available_alignment == 0
            || !available_alignment.is_power_of_two()
        {
            return Err(TensorStorageValidationError::InvalidAlignment {
                value: self.value,
                required_bytes: self.alignment_bytes,
                available_bytes: available_alignment,
            });
        }
        if available_alignment < self.alignment_bytes
            || !available_alignment.is_multiple_of(self.alignment_bytes)
        {
            return Err(TensorStorageValidationError::InsufficientAlignment {
                value: self.value,
                required_bytes: self.alignment_bytes,
                available_bytes: available_alignment,
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

pub const fn dense_scalar_layout(scalar_type: PcuScalarType) -> Option<(usize, usize)> {
    match scalar_type {
        PcuScalarType::I8 | PcuScalarType::U8 | PcuScalarType::F8E4M3FN | PcuScalarType::F8E5M2 => {
            Some((1, 1))
        }
        PcuScalarType::I16 | PcuScalarType::U16 | PcuScalarType::F16 | PcuScalarType::BF16 => {
            Some((2, 2))
        }
        PcuScalarType::I32 | PcuScalarType::U32 | PcuScalarType::F32 => Some((4, 4)),
        PcuScalarType::I64 | PcuScalarType::U64 | PcuScalarType::F64 => Some((8, 8)),
        PcuScalarType::I128 | PcuScalarType::U128 => Some((16, 16)),
        PcuScalarType::F128 => Some((16, 8)),
        PcuScalarType::I256 | PcuScalarType::U256 | PcuScalarType::F256 => Some((32, 8)),
        PcuScalarType::I512 | PcuScalarType::U512 => Some((64, 8)),
        // These scalar vocabulary entries have no agreed byte-addressable dense encoding.
        PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4 => None,
    }
}

fn align_up(bytes: usize, alignment: usize) -> Option<usize> {
    debug_assert!(alignment.is_power_of_two());
    bytes
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

pub fn node_output_bytes(shape: &[usize], scalar_type: PcuScalarType) -> Option<usize> {
    let (element_bytes, _) = dense_scalar_layout(scalar_type)?;
    shape.iter().try_fold(element_bytes, |bytes, &dimension| {
        bytes.checked_mul(dimension)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_scalar_layout_uses_byte_addressable_abi_only() {
        assert_eq!(dense_scalar_layout(PcuScalarType::I8), Some((1, 1)));
        assert_eq!(dense_scalar_layout(PcuScalarType::U16), Some((2, 2)));
        assert_eq!(dense_scalar_layout(PcuScalarType::BF16), Some((2, 2)));
        assert_eq!(dense_scalar_layout(PcuScalarType::F32), Some((4, 4)));
        assert_eq!(dense_scalar_layout(PcuScalarType::F64), Some((8, 8)));
        assert_eq!(dense_scalar_layout(PcuScalarType::Bool), None);
        assert_eq!(dense_scalar_layout(PcuScalarType::I4), None);
        assert_eq!(dense_scalar_layout(PcuScalarType::U4), None);
        assert_eq!(node_output_bytes(&[2, 3], PcuScalarType::F64), Some(48));
        assert_eq!(
            node_output_bytes(&[2, usize::MAX], PcuScalarType::F64),
            None
        );
    }

    #[test]
    fn scratch_slot_upgrade_preserves_larger_alignment_and_capacity() {
        let liveness = [
            TensorValueLiveness {
                value: ValueId {
                    graph_id: 1,
                    index: 0,
                },
                scalar_type: PcuScalarType::F32,
                output_bytes: Some(12),
                alignment_bytes: Some(4),
                first_live_node: 0,
                last_live_node: 0,
            },
            TensorValueLiveness {
                value: ValueId {
                    graph_id: 1,
                    index: 1,
                },
                scalar_type: PcuScalarType::F64,
                output_bytes: Some(8),
                alignment_bytes: Some(8),
                first_live_node: 1,
                last_live_node: 1,
            },
        ];
        let plan =
            plan_scratch_storage(&liveness, &[liveness[0].value, liveness[1].value]).unwrap();

        assert_eq!(plan.slot_for(liveness[0].value), Some(0));
        assert_eq!(plan.slot_for(liveness[1].value), Some(0));
        assert_eq!(
            plan.slots(),
            &[TensorScratchStorageSlot {
                capacity_bytes: 16,
                alignment_bytes: 8,
            }]
        );
        assert_eq!(plan.total_bytes(), 16);
    }

    #[test]
    fn scratch_planner_rejects_scalar_without_dense_layout() {
        let life = TensorValueLiveness {
            value: ValueId {
                graph_id: 1,
                index: 0,
            },
            scalar_type: PcuScalarType::I4,
            output_bytes: None,
            alignment_bytes: None,
            first_live_node: 0,
            last_live_node: 0,
        };

        assert_eq!(
            plan_scratch_storage(&[life], &[life.value]),
            Err(super::super::TensorError::UnsupportedScalarType {
                value: life.value,
                scalar_type: PcuScalarType::I4,
            })
        );
    }

    #[test]
    fn scratch_planner_rejects_invalid_alignment_facts() {
        let life = TensorValueLiveness {
            value: ValueId {
                graph_id: 1,
                index: 0,
            },
            scalar_type: PcuScalarType::F32,
            output_bytes: Some(4),
            alignment_bytes: Some(6),
            first_live_node: 0,
            last_live_node: 0,
        };

        assert_eq!(
            plan_scratch_storage(&[life], &[life.value]),
            Err(super::super::TensorError::InvalidStorageAlignment {
                value: life.value,
                alignment_bytes: 6,
            })
        );
    }
}

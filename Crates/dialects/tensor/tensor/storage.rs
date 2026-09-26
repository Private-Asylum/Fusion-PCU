//! Backend-neutral tensor storage requirements and overlap validation.

use fusion_pcu::{
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
    Overlapping { left: ValueId, right: ValueId },
    UnknownOverlap { left: ValueId, right: ValueId },
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

pub fn node_output_bytes(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(std::mem::size_of::<f32>(), |bytes, &dimension| {
            bytes.checked_mul(dimension)
        })
}

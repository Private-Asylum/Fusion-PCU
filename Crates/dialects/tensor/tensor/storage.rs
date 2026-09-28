//! Backend-neutral tensor storage requirements and overlap validation.

#[rustfmt::skip]
use fusion_pcu::{
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
        .try_fold(std::mem::size_of::<f32>(), |bytes, &dimension| {
            bytes.checked_mul(dimension)
        })
}

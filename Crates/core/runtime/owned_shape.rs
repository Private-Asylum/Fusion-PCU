//! Small-rank owned shape metadata for device tensors.

use alloc::vec::Vec;

/// Owned dense dimensions with allocation-free storage for common tensor ranks.
///
/// Ranks through four fit inline, covering scalar, vector, matrix, and batched 4D shapes; higher
/// ranks spill to an owned vector. Borrowed shapes can be copied directly with
/// [`PcuOwnedShape::from_slice`], avoiding an intermediate heap allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PcuOwnedShape(PcuOwnedShapeStorage);

#[derive(Clone, Debug, PartialEq, Eq)]
enum PcuOwnedShapeStorage {
    Inline { len: usize, dimensions: [usize; 4] },
    Heap(Vec<usize>),
}

impl PcuOwnedShape {
    /// Number of dimensions kept inline before storage spills to the heap.
    pub const INLINE_CAPACITY: usize = 4;

    /// Copies a borrowed shape inline when its rank is at most four, otherwise into owned storage.
    #[must_use]
    pub fn from_slice(dimensions: &[usize]) -> Self {
        if dimensions.len() <= Self::INLINE_CAPACITY {
            let mut inline = [0; Self::INLINE_CAPACITY];
            inline[..dimensions.len()].copy_from_slice(dimensions);
            Self(PcuOwnedShapeStorage::Inline {
                len: dimensions.len(),
                dimensions: inline,
            })
        } else {
            Self(PcuOwnedShapeStorage::Heap(Vec::from(dimensions)))
        }
    }

    /// Dense dimensions as a borrowed slice.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        match &self.0 {
            PcuOwnedShapeStorage::Inline { len, dimensions } => &dimensions[..*len],
            PcuOwnedShapeStorage::Heap(dimensions) => dimensions,
        }
    }

    #[cfg(test)]
    const fn is_inline(&self) -> bool {
        matches!(self.0, PcuOwnedShapeStorage::Inline { .. })
    }
}

impl<const N: usize> From<[usize; N]> for PcuOwnedShape {
    fn from(dimensions: [usize; N]) -> Self {
        Self::from_slice(&dimensions)
    }
}

impl From<&[usize]> for PcuOwnedShape {
    fn from(dimensions: &[usize]) -> Self {
        Self::from_slice(dimensions)
    }
}

impl<const N: usize> From<&[usize; N]> for PcuOwnedShape {
    fn from(dimensions: &[usize; N]) -> Self {
        Self::from_slice(dimensions)
    }
}

impl From<Vec<usize>> for PcuOwnedShape {
    fn from(dimensions: Vec<usize>) -> Self {
        if dimensions.len() <= Self::INLINE_CAPACITY {
            Self::from_slice(&dimensions)
        } else {
            Self(PcuOwnedShapeStorage::Heap(dimensions))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_shape_keeps_common_ranks_inline_and_spills_without_changing_slice_view() {
        for rank in 0..=PcuOwnedShape::INLINE_CAPACITY {
            let dimensions = [2, 3, 4, 5];
            let shape = PcuOwnedShape::from_slice(&dimensions[..rank]);
            assert!(shape.is_inline());
            assert_eq!(shape.as_slice(), &dimensions[..rank]);
            assert_eq!(shape.clone(), shape);
        }

        let dimensions = [2, 3, 4, 5, 6];
        let shape = PcuOwnedShape::from_slice(&dimensions);
        assert!(!shape.is_inline());
        assert_eq!(shape.as_slice(), &dimensions);
        let cloned = shape.clone();
        assert_eq!(cloned, shape);
        assert_ne!(cloned.as_slice().as_ptr(), shape.as_slice().as_ptr());
    }

    #[test]
    fn owned_shape_retains_spilled_vec_and_compacts_small_vec() {
        let spilled = alloc::vec![2, 3, 4, 5, 6];
        let allocation = spilled.as_ptr();
        let shape = PcuOwnedShape::from(spilled);
        assert!(!shape.is_inline());
        assert_eq!(shape.as_slice().as_ptr(), allocation);

        let small = PcuOwnedShape::from(alloc::vec![2, 3]);
        assert!(small.is_inline());
        assert_eq!(small.as_slice(), &[2, 3]);
    }
}

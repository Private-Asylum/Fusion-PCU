//! Owned, shaped device-resident tensor storage.

use core::fmt;

#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuDeviceArgument,
    PcuDeviceBuffer,
    PcuOwnedShape,
    PcuScalar,
};

/// Failure while pairing dense tensor shape metadata with a typed device buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcuDeviceTensorError {
    /// The product of the tensor dimensions does not fit in `usize`.
    ShapeOverflow,
    /// The buffer's logical element count differs from the dense shape's element count.
    ElementCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for PcuDeviceTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow => f.write_str("device tensor shape element count overflows usize"),
            Self::ElementCountMismatch { expected, actual } => write!(
                f,
                "device tensor shape requires {expected} elements, but its buffer declares {actual}"
            ),
        }
    }
}

impl core::error::Error for PcuDeviceTensorError {}

/// An owned dense tensor whose values remain in backend-owned device storage.
///
/// The shape uses row-major dense semantics, matching the tensor dialect. The buffer is moved
/// into this value and is not cloneable, so ownership transfer does not create an implicit alias.
pub struct PcuDeviceTensor<T: PcuScalar, R> {
    shape: PcuOwnedShape,
    buffer: PcuDeviceBuffer<T, R>,
}

impl<T: PcuScalar, R> PcuDeviceTensor<T, R> {
    /// Pairs a dense shape with a buffer whose logical element count exactly matches it.
    ///
    /// # Errors
    ///
    /// Returns [`PcuDeviceTensorError::ShapeOverflow`] for an overflowing dimension product or
    /// [`PcuDeviceTensorError::ElementCountMismatch`] when the buffer length differs.
    pub fn new(
        shape: impl Into<PcuOwnedShape>,
        buffer: PcuDeviceBuffer<T, R>,
    ) -> Result<Self, PcuDeviceTensorError> {
        let shape = shape.into();
        let expected = shape
            .as_slice()
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension));
        let expected = expected.ok_or(PcuDeviceTensorError::ShapeOverflow)?;
        if expected != buffer.len() {
            return Err(PcuDeviceTensorError::ElementCountMismatch {
                expected,
                actual: buffer.len(),
            });
        }
        Ok(Self { shape, buffer })
    }

    /// Dense row-major dimensions for this tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// Typed device buffer backing this tensor.
    #[must_use]
    pub const fn buffer(&self) -> &PcuDeviceBuffer<T, R> {
        &self.buffer
    }

    /// Exclusive backend storage for provider updates without changing the logical shape/type.
    ///
    /// Physical replacements are revalidated by execution; this does not establish initialization.
    #[must_use]
    pub const fn resource_mut(&mut self) -> &mut R {
        self.buffer.resource_mut()
    }

    /// Creates a read-only typed device argument while preserving this tensor's shape metadata.
    #[must_use]
    pub const fn read_argument(&self, target: PcuBindingRef) -> PcuDeviceArgument<'_, R> {
        PcuDeviceArgument::read(target, &self.buffer)
    }

    /// Creates an exclusive read/write device argument while preserving this tensor's shape
    /// metadata.
    #[must_use]
    pub const fn read_write_argument(&mut self, target: PcuBindingRef) -> PcuDeviceArgument<'_, R> {
        PcuDeviceArgument::read_write(target, &mut self.buffer)
    }

    /// Recovers the device buffer, consuming its shape wrapper.
    #[must_use]
    pub fn into_buffer(self) -> PcuDeviceBuffer<T, R> {
        self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn device_tensor_checks_dense_shape_against_logical_buffer_length() {
        let buffer = PcuDeviceBuffer::<f32, _>::new([0_u8; 24], 6);
        let mut tensor = PcuDeviceTensor::new([2, 3], buffer).expect("matching dense shape");
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.buffer().len(), 6);
        {
            let argument = tensor.read_argument(PcuBindingRef::new(0, 0));
            assert_eq!(argument.elements(), 6);
            assert_eq!(argument.access(), crate::PcuBindingAccess::ReadOnly);
        }
        {
            let argument = tensor.read_write_argument(PcuBindingRef::new(0, 0));
            assert_eq!(argument.elements(), 6);
            assert_eq!(argument.access(), crate::PcuBindingAccess::ReadWrite);
        }
        assert_eq!(tensor.shape(), &[2, 3]);

        let short = PcuDeviceBuffer::<f32, _>::new([0_u8; 20], 5);
        assert!(matches!(
            PcuDeviceTensor::new([2, 3], short),
            Err(PcuDeviceTensorError::ElementCountMismatch {
                expected: 6,
                actual: 5,
            })
        ));
    }

    #[test]
    fn device_tensor_rejects_shape_product_overflow() {
        let buffer = PcuDeviceBuffer::<u8, [u8; 0]>::new([], 0);
        assert!(matches!(
            PcuDeviceTensor::new([usize::MAX, 2], buffer),
            Err(PcuDeviceTensorError::ShapeOverflow)
        ));
    }

    #[test]
    fn device_tensor_accepts_borrowed_inline_and_spilled_shapes() {
        let scalar = PcuDeviceTensor::<u8, [u8; 1]>::new(
            PcuOwnedShape::from_slice(&[]),
            PcuDeviceBuffer::new([0], 1),
        )
        .expect("rank-zero scalar has one element");
        assert_eq!(scalar.shape(), &[]);

        let small_dimensions = [2, 3];
        let small = PcuDeviceTensor::<u8, [u8; 6]>::new(
            PcuOwnedShape::from_slice(&small_dimensions),
            PcuDeviceBuffer::new([0; 6], 6),
        )
        .expect("borrowed low-rank shape");
        assert_eq!(small.shape(), &[2, 3]);

        let large_dimensions = [1, 1, 1, 1, 6];
        let large = PcuDeviceTensor::<u8, [u8; 6]>::new(
            PcuOwnedShape::from_slice(&large_dimensions),
            PcuDeviceBuffer::new([0; 6], 6),
        )
        .expect("borrowed high-rank shape");
        assert_eq!(large.shape(), &large_dimensions);

        let empty = PcuDeviceTensor::<u8, [u8; 0]>::new([2, 0, 3], PcuDeviceBuffer::new([], 0))
            .expect("zero extent has zero elements");
        assert_eq!(empty.shape(), &[2, 0, 3]);
    }

    #[test]
    fn device_tensor_drops_buffer_when_shape_validation_fails() {
        let drops = Arc::new(AtomicUsize::new(0));
        let resource = DropProbe(Arc::clone(&drops));
        let buffer = PcuDeviceBuffer::<u8, _>::new(resource, 1);

        assert!(matches!(
            PcuDeviceTensor::new([2], buffer),
            Err(PcuDeviceTensorError::ElementCountMismatch { .. })
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

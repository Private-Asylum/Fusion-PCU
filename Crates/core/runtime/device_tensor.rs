//! Owned, shaped device-resident tensor storage.

use alloc::vec::Vec;
use core::fmt;

#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuDeviceArgument,
    PcuDeviceBuffer,
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
    shape: Vec<usize>,
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
        shape: impl Into<Vec<usize>>,
        buffer: PcuDeviceBuffer<T, R>,
    ) -> Result<Self, PcuDeviceTensorError> {
        let shape = shape.into();
        let expected = shape
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
        &self.shape
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
}

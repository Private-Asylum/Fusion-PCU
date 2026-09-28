//! Typed device ownership and reusable resident kernel calls.

use core::marker::PhantomData;

#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuScalar,
    PcuScalarType,
};

/// An owned backend resource with a scalar element type and logical length.
///
/// This carries device data, not a host slice or an implicit result cache. Backends validate its
/// real allocation, device identity and byte extent before use. It deliberately does not implement
/// Clone: ordinary Rust borrowing expresses read-only and exclusive mutable kernel arguments.
pub struct PcuDeviceBuffer<T: PcuScalar, R> {
    resource: R,
    len: usize,
    scalar: PhantomData<T>,
}

impl<T: PcuScalar, R> PcuDeviceBuffer<T, R> {
    /// Wraps owned backend storage. Declared length is checked against actual storage at execution.
    #[must_use]
    pub const fn new(resource: R, len: usize) -> Self {
        Self {
            resource,
            len,
            scalar: PhantomData,
        }
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Backend storage, exposed for advanced integrations and validated again on use.
    #[must_use]
    pub const fn resource(&self) -> &R {
        &self.resource
    }
    /// Exclusive backend storage for provider updates; execution revalidates any replacement.
    #[must_use]
    pub const fn resource_mut(&mut self) -> &mut R {
        &mut self.resource
    }
    /// Recovers the owned backend storage.
    #[must_use]
    pub fn into_resource(self) -> R {
        self.resource
    }
}

/// A typed resource borrow for a synchronous device-resident call.
pub struct PcuDeviceArgument<'a, R> {
    target: PcuBindingRef,
    scalar: PcuScalarType,
    access: PcuBindingAccess,
    elements: usize,
    resource: &'a R,
}

impl<'a, R> PcuDeviceArgument<'a, R> {
    #[must_use]
    pub const fn read<T: PcuScalar>(
        target: PcuBindingRef,
        buffer: &'a PcuDeviceBuffer<T, R>,
    ) -> Self {
        Self {
            target,
            scalar: T::TYPE,
            access: PcuBindingAccess::ReadOnly,
            elements: buffer.len,
            resource: &buffer.resource,
        }
    }
    #[must_use]
    pub const fn read_write<T: PcuScalar>(
        target: PcuBindingRef,
        buffer: &'a mut PcuDeviceBuffer<T, R>,
    ) -> Self {
        Self {
            target,
            scalar: T::TYPE,
            access: PcuBindingAccess::ReadWrite,
            elements: buffer.len,
            resource: &buffer.resource,
        }
    }
    #[must_use]
    pub const fn target(&self) -> PcuBindingRef {
        self.target
    }
    #[must_use]
    pub const fn scalar(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn access(&self) -> PcuBindingAccess {
        self.access
    }
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }
    #[must_use]
    pub const fn resource(&self) -> &R {
        self.resource
    }
}

/// Backend-neutral compilation for typed device-resident calls.
pub trait PcuDeviceKernelBackend {
    type Resource;
    type Prepared: PcuPreparedDeviceKernel<Resource = Self::Resource>;
    type Error;
    /// Captures an executable independently of the temporary source IR.
    ///
    /// # Errors
    /// Returns unsupported capabilities, validation or compilation errors.
    fn prepare_device_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error>;
}

/// Reusable executable operating directly on typed device data.
pub trait PcuPreparedDeviceKernel {
    type Resource;
    type Error;
    /// Validates and completes a device-resident call without implicit host transfers.
    ///
    /// Every call validates actual storage, type, access, extent and device identity. Retain device
    /// resources through completion or quarantine. Complete before returning exclusive borrows;
    /// a subsequent kernel may then safely consume the updated buffer through ordinary Rust code.
    ///
    /// # Errors
    /// Returns admission, submission, completion or checked-operation errors.
    /// A failed kernel may have changed mutable buffers before its fault; this is not a
    /// transactional rollback contract. Uncertain completion must prevent unsafe resource reuse.
    fn call(
        &mut self,
        arguments: &mut [PcuDeviceArgument<'_, Self::Resource>],
    ) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_borrows_describe_ownership_without_copying_storage() {
        let mut buffer = PcuDeviceBuffer::<u64, _>::new([9_u8; 16], 2);
        let argument = PcuDeviceArgument::read(PcuBindingRef::new(0, 3), &buffer);
        assert_eq!(argument.scalar(), PcuScalarType::U64);
        assert_eq!(argument.elements(), 2);
        assert_eq!(argument.access(), PcuBindingAccess::ReadOnly);
        assert!(core::ptr::eq(argument.resource(), buffer.resource()));
        let argument = PcuDeviceArgument::read_write(PcuBindingRef::new(0, 3), &mut buffer);
        assert_eq!(argument.access(), PcuBindingAccess::ReadWrite);
        assert_eq!(buffer.into_resource(), [9; 16]);
    }
}

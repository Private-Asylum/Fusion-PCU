//! Borrowed actual Vulkan byte owners; no public handle or forged device address is accepted.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuHostArgument,
    PcuScalar,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    ffi::VulkanOwnedBuffer,
    PcuVulkanOwnedBuffer,
};

/// One exact host or native scalar argument, borrowed until synchronous terminal completion.
///
/// Native variants can only be constructed by a typed owned buffer. Their private payload
/// retains its actual logical-device root; the lifetime prevents drop/reallocation during a call.
/// Construction authorizes neither arithmetic nor session migration.
pub struct PcuVulkanArgument<'a> {
    pub(crate) target: PcuBindingRef,
    pub(crate) scalar: PcuScalarType,
    pub(crate) kind: Kind<'a>,
}
pub enum Kind<'a> {
    Host(PcuHostArgument<'a>),
    Read(&'a VulkanOwnedBuffer),
    Write(&'a mut VulkanOwnedBuffer),
}
impl<'a> PcuVulkanArgument<'a> {
    /// Borrows the caller's exact typed host declaration; no upload occurs at construction.
    #[must_use]
    pub const fn host(argument: PcuHostArgument<'a>) -> Self {
        Self {
            target: argument.target(),
            scalar: argument.scalar(),
            kind: Kind::Host(argument),
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
        match &self.kind {
            Kind::Host(argument) => argument.access(),
            Kind::Read(_) => PcuBindingAccess::ReadOnly,
            Kind::Write(_) => PcuBindingAccess::ReadWrite,
        }
    }
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        match &self.kind {
            Kind::Host(argument) => argument.bytes().len(),
            Kind::Read(buffer) => buffer.byte_len(),
            Kind::Write(buffer) => buffer.byte_len(),
        }
    }
}
impl<T: PcuScalar> PcuVulkanOwnedBuffer<T> {
    /// Borrows initialized native scalar storage for a checked same-session prepared call.
    /// No payload transfer or SDK query is performed.
    #[must_use]
    pub const fn read_argument(&self, target: PcuBindingRef) -> PcuVulkanArgument<'_> {
        PcuVulkanArgument {
            target,
            scalar: T::TYPE,
            kind: Kind::Read(self.native()),
        }
    }
    /// Exclusively borrows native output storage. A prepared mixed call stages private outputs
    /// and publishes only its complete logical prefix after the complete fatal-status scan.
    #[must_use]
    pub const fn write_argument(&mut self, target: PcuBindingRef) -> PcuVulkanArgument<'_> {
        PcuVulkanArgument {
            target,
            scalar: T::TYPE,
            kind: Kind::Write(self.native_mut()),
        }
    }
}

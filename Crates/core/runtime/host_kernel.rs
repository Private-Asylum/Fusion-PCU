//! Typed host borrows for synchronous, reusable kernel execution.
//!
//! This describes a host boundary, not device selection or implicit CPU fallback. Native host
//! byte order is explicit: implementations must convert it or reject an unsupported host order.

#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuScalar,
    PcuScalarType,
};

enum HostBorrow<'a> {
    Read(&'a [u8]),
    ReadWrite(&'a mut [u8]),
}

/// One typed resource borrow for a synchronous kernel call.
///
/// Constructors accept only sealed PCU scalars, whose initialized representation has no padding
/// and accepts every bit pattern. A mutable borrow retains Rust's exclusive access for the call.
/// Host bytes are not an asynchronous device pointer and must not escape this borrow.
pub struct PcuHostArgument<'a> {
    target: PcuBindingRef,
    scalar: PcuScalarType,
    memory: HostBorrow<'a>,
}

impl<'a> PcuHostArgument<'a> {
    /// Borrows one scalar as a read-only resource without allocating or encoding a copy.
    #[must_use]
    pub const fn read_scalar<T: PcuScalar>(target: PcuBindingRef, value: &'a T) -> Self {
        Self::read(target, core::slice::from_ref(value))
    }

    /// Borrows one scalar as a read/write resource without allocating or encoding a copy.
    ///
    /// Implementations must upload the initial value when needed and preserve it if unwritten.
    #[must_use]
    pub const fn read_write_scalar<T: PcuScalar>(target: PcuBindingRef, value: &'a mut T) -> Self {
        Self::read_write(target, core::slice::from_mut(value))
    }

    /// Borrows a scalar slice as a read-only resource without allocating or encoding a copy.
    #[must_use]
    #[allow(unsafe_code)] // Sealed PcuScalar implementations are padding-free and fully initialized.
    pub const fn read<T: PcuScalar>(target: PcuBindingRef, values: &'a [T]) -> Self {
        // SAFETY: Every sealed scalar has a padding-free primitive, encoded-bit or limb representation.
        // The byte view covers precisely the initialized slice and cannot outlive its shared borrow.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        };
        Self {
            target,
            scalar: T::TYPE,
            memory: HostBorrow::Read(bytes),
        }
    }

    /// Borrows a scalar slice as a read/write resource without allocating or encoding a copy.
    ///
    /// Implementations must upload initial contents when needed and preserve unwritten elements.
    #[must_use]
    #[allow(unsafe_code)] // All bit patterns are valid for every sealed PcuScalar implementation.
    pub const fn read_write<T: PcuScalar>(target: PcuBindingRef, values: &'a mut [T]) -> Self {
        let len = core::mem::size_of_val(values);
        // SAFETY: The same padding-free representation law applies as in read. All possible device
        // result bits are valid values of T; exclusive access is retained for this view's lifetime.
        let bytes =
            unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len) };
        Self {
            target,
            scalar: T::TYPE,
            memory: HostBorrow::ReadWrite(bytes),
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
        match self.memory {
            HostBorrow::Read(_) => PcuBindingAccess::ReadOnly,
            HostBorrow::ReadWrite(_) => PcuBindingAccess::ReadWrite,
        }
    }
    /// Initialized native-endian bytes, including the initial contents of a mutable resource.
    #[must_use]
    pub const fn bytes(&self) -> &[u8] {
        match &self.memory {
            HostBorrow::Read(bytes) => bytes,
            HostBorrow::ReadWrite(bytes) => bytes,
        }
    }
    /// Exclusive result bytes, available only for a mutable resource.
    #[must_use]
    pub const fn bytes_mut(&mut self) -> Option<&mut [u8]> {
        match &mut self.memory {
            HostBorrow::Read(_) => None,
            HostBorrow::ReadWrite(bytes) => Some(bytes),
        }
    }
}

/// Backend-neutral preparation of an owned executable for typed host calls.
///
/// The result must own compiled state and schemas independently of the temporary IR. Runtime
/// selection remains explicit in the backend value; preparation must never silently select CPU.
pub trait PcuHostKernelBackend {
    type Prepared: PcuPreparedHostKernel;
    type Error;
    /// Compiles and captures a reusable executable; this is outside the warm call path.
    ///
    /// # Errors
    /// Returns validation, capability, compilation, or device errors.
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error>;
}

/// A reusable executable with normal Rust host-borrow semantics.
pub trait PcuPreparedHostKernel {
    type Error;
    /// Executes synchronously using fresh contents from all supplied resources.
    ///
    /// Validate coverage, types, access and extents before launching. Reuse owned storage where
    /// legal; supply current mutable contents whenever the kernel can read them, preserving
    /// unwritten elements. A proven complete writer may omit its incoming copy and read back
    /// only its initialized prefix, leaving untouched host tails unchanged. Complete device work
    /// before exposing results or releasing host borrows. No borrowed host pointer may remain in
    /// flight after return, including errors. Pending device owners must survive completion or
    /// quarantine. Changed input contents must never be mistaken for cached results.
    ///
    /// Publication is a transaction across all supplied host outputs. Complete all fallible
    /// transfers, status validation and sibling preflight before changing caller bytes. A backend
    /// whose readback can fail after writing must use retained private host storage; a backend
    /// with fully validated mapped output may commit with infallible copies directly. Do not
    /// impose an additional copy where the existing mechanism already proves this contract.
    /// Preserve unwritten tails. This host transaction does not authorize restoring possibly
    /// written resident storage: escaped device owners retain their separate discard/quarantine
    /// law, and are not the host borrows described by this trait.
    ///
    /// # Errors
    /// Returns admission, allocation, transfer, execution or checked-operation errors. A fatal
    /// error preserves every supplied host output's pre-call bytes, including errors in later
    /// sibling readbacks. A completed, explicitly requested Clamp may instead publish all useful
    /// outputs and return its observable recovered range error. Failure to finish that publication
    /// is fatal; a recovered arithmetic notice never excuses a partial host commit.
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[rustfmt::skip]
    use crate::{
        PcuBf16Bits,
        PcuF16Bits,
    };

    #[test]
    fn shared_and_mutable_views_preserve_bits_and_access() {
        let input = [f64::from_bits(0x7ff0_0000_0000_0001), -0.0];
        let mut read = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input);
        assert_eq!(read.scalar(), PcuScalarType::F64);
        assert_eq!(read.access(), PcuBindingAccess::ReadOnly);
        assert_eq!(&read.bytes()[..8], &input[0].to_ne_bytes());
        assert!(read.bytes_mut().is_none());
        let mut output = [0_u64; 2];
        {
            let mut argument = PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output);
            assert_eq!(argument.access(), PcuBindingAccess::ReadWrite);
            argument.bytes_mut().expect("mutable")[..8].copy_from_slice(&u64::MAX.to_ne_bytes());
        }
        assert_eq!(output, [u64::MAX, 0]);
    }

    #[test]
    fn half_and_empty_views_are_valid() {
        let halves = [PcuF16Bits::from_bits(0x7c01)];
        assert_eq!(
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &halves).bytes(),
            &0x7c01_u16.to_ne_bytes()
        );
        let mut halves = [PcuBf16Bits::from_bits(0)];
        PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut halves)
            .bytes_mut()
            .expect("mutable")
            .copy_from_slice(&0xffff_u16.to_ne_bytes());
        assert_eq!(halves[0].to_bits(), 0xffff);
        let mut empty: [i8; 0] = [];
        assert!(
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut empty)
                .bytes()
                .is_empty()
        );
    }

    #[test]
    fn scalar_views_preserve_exact_type_access_and_borrowed_bits() {
        let seed = f64::from_bits(0x7ff0_0000_0000_0001);
        let mut read = PcuHostArgument::read_scalar(PcuBindingRef::new(3, 7), &seed);
        assert_eq!(read.target(), PcuBindingRef::new(3, 7));
        assert_eq!(read.scalar(), PcuScalarType::F64);
        assert_eq!(read.access(), PcuBindingAccess::ReadOnly);
        assert_eq!(read.bytes(), &seed.to_ne_bytes());
        assert!(read.bytes_mut().is_none());

        let mut result = 0_u64;
        {
            let mut argument =
                PcuHostArgument::read_write_scalar(PcuBindingRef::new(3, 8), &mut result);
            assert_eq!(argument.target(), PcuBindingRef::new(3, 8));
            assert_eq!(argument.scalar(), PcuScalarType::U64);
            assert_eq!(argument.access(), PcuBindingAccess::ReadWrite);
            assert_eq!(argument.bytes(), &0_u64.to_ne_bytes());
            argument
                .bytes_mut()
                .expect("mutable scalar")
                .copy_from_slice(&0xfedc_ba98_7654_3210_u64.to_ne_bytes());
        }
        // The exclusive borrow ends with the argument, making the updated scalar available.
        assert_eq!(result, 0xfedc_ba98_7654_3210);
    }
}

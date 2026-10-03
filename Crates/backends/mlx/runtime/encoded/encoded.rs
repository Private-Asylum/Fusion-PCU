//! Immutable logical encodings backed by exact verified MLX `UInt` carriers, distinct from F32 matrices.
#[rustfmt::skip]
use std::{rc::Rc,cell::RefCell,ops::Deref};
#[rustfmt::skip]
use fusion_pcu::{PcuScalar,PcuScalarType,PcuBindingRef,PcuHostArgument,PcuExecutionFault};
#[rustfmt::skip]
use crate::{ffi,MlxError};
use super::MlxSession;
#[path = "prefix/prefix.rs"]
mod prefix;
pub use prefix::MlxPreparedEncodedPrefix;

/// Immutable byte-aligned encoding array with a retained exact MLX session and official backing.
///
/// Logical dtype/count are independent from physical UInt8/UInt16/UInt32 limb carriers. Clones share ownership;
/// they do not create a device session, copy data or establish cross-provider interoperability.
#[derive(Clone)]
pub struct MlxEncodedArray {
    session: MlxSession,
    native: Rc<EncodedBacking>,
}
pub(super) struct EncodedBacking {
    native: ffi::EncodedArray,
    views: RefCell<Vec<([usize; 2], Rc<super::Array>)>>,
}
impl Deref for EncodedBacking {
    type Target = ffi::EncodedArray;
    fn deref(&self) -> &Self::Target {
        &self.native
    }
}
#[path = "view/view.rs"]
mod view;
impl MlxEncodedArray {
    #[must_use]
    pub fn scalar_type(&self) -> PcuScalarType {
        self.native.scalar()
    }
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.native.count()
    }
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.native.byte_len()
    }
    /// The exact retained session; borrowing it never opens or rebinds a device.
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        &self.session
    }
    /// Exact retained runtime/stream affinity; matching GPU index or logical dtype is insufficient.
    #[must_use]
    pub fn same_session(&self, session: &MlxSession) -> bool {
        self.session.same_session(session)
    }
    /// Checks the retained session quarantine before borrowing or publishing this backing.
    ///
    /// # Errors
    /// Returns completion uncertainty when actual native work/session owners are quarantined.
    pub fn validate_access_available(&self) -> Result<(), MlxError> {
        self.session.validate_access_available()
    }
    /// Reads a verified terminal dense carrier into the exact typed prefix, preserving the tail.
    ///
    /// # Errors
    /// Returns logical scalar/extent mismatch, quarantine or native materialization failure.
    /// The destination remains unchanged unless all terminal and native cleanup checks succeed.
    pub fn read_into<T: PcuScalar>(&self, output: &mut [T]) -> Result<(), MlxError> {
        if T::TYPE != self.scalar_type() {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        if output.len() < self.element_count() {
            return Err(MlxError::InvalidExtent);
        }
        let mut destination = PcuHostArgument::read_write(
            PcuBindingRef::new(0, 0),
            &mut output[..self.element_count()],
        );
        self.native
            .read(destination.bytes_mut().ok_or(MlxError::InvalidExtent)?)
    }
    /// Reads the exact logical span into initialized bytes after terminal native cleanup.
    /// The caller's larger host destination must select its admitted prefix explicitly.
    ///
    /// # Errors
    /// Rejects a nonexact span, quarantine or native failure without changing the destination.
    pub fn read_bytes_into(&self, output: &mut [u8]) -> Result<(), MlxError> {
        if output.len() != self.byte_len() {
            return Err(MlxError::InvalidExtent);
        }
        self.native.read(output)
    }
    /// Consumes this completed handle, checking the final native holder release when unique.
    /// A shared handle releases only its Rust reference; its other immutable owners stay live.
    /// Unique cached F32 views are not silently freed through this bounded private-owner seam.
    ///
    /// # Errors
    /// Returns quarantine, unsupported cached-view cleanup or a failed official holder release.
    /// The consumed handle cannot be reused. Unknown sessions retain their native resources.
    pub fn release(self) -> Result<(), MlxError> {
        self.validate_access_available()?;
        match Rc::try_unwrap(self.native) {
            Ok(backing) => {
                if !backing.views.borrow().is_empty() {
                    return Err(MlxError::InvalidRequest(
                        "checked encoded release does not admit cached native views".into(),
                    ));
                }
                backing.native.release()
            }
            Err(shared) => {
                // Another immutable owner retains the exact official holder, so consuming
                // this Rc reference cannot invoke any foreign free between publications.
                drop(shared);
                Ok(())
            }
        }
    }
    pub(crate) fn native(&self) -> &ffi::EncodedArray {
        &self.native
    }
}
/// A fully completed private output and optional observable range recovery.
///
/// Fatal or unknown execution returns no owner. Callers decide whether to publish this new
/// immutable value; no preexisting owner's backing has been modified by this completion.
pub struct MlxEncodedCompletion {
    output: MlxEncodedArray,
    recovered: Option<PcuExecutionFault>,
}
impl MlxEncodedCompletion {
    #[must_use]
    pub const fn output(&self) -> &MlxEncodedArray {
        &self.output
    }
    #[must_use]
    pub const fn recovered_fault(&self) -> Option<PcuExecutionFault> {
        self.recovered
    }
    #[must_use]
    pub fn into_parts(self) -> (MlxEncodedArray, Option<PcuExecutionFault>) {
        (self.output, self.recovered)
    }
}
impl MlxSession {
    /// Copies initialized byte-aligned logical encodings into verified official `UInt` carriers.
    /// All22 byte-aligned sealed scalar representations transport exact bits; Bool/I4/U4 reject.
    /// This does not authorize arithmetic or reinterpret a native F32 MLX tensor backing.
    ///
    /// # Errors
    /// Rejects unsupported scalar or extent before native upload; returns quarantine/native failure.
    pub fn upload_encoded<T: PcuScalar>(&self, input: &[T]) -> Result<MlxEncodedArray, MlxError> {
        let source = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
        let native =
            ffi::EncodedArray::upload(&self.0.native, T::TYPE, input.len(), source.bytes())?;
        Ok(self.wrap_encoded(native))
    }
    pub(crate) fn upload_encoded_bytes(
        &self,
        scalar: PcuScalarType,
        count: usize,
        input: &[u8],
    ) -> Result<MlxEncodedArray, MlxError> {
        let native = ffi::EncodedArray::upload(&self.0.native, scalar, count, input)?;
        Ok(self.wrap_encoded(native))
    }
    pub(crate) fn upload_carrier_native(
        &self,
        scalar: PcuScalarType,
        count: usize,
        input: &[u8],
    ) -> Result<ffi::EncodedArray, MlxError> {
        ffi::EncodedArray::upload(&self.0.native, scalar, count, input)
    }
    pub(crate) fn wrap_encoded(&self, native: ffi::EncodedArray) -> MlxEncodedArray {
        MlxEncodedArray {
            session: self.clone(),
            native: Rc::new(EncodedBacking {
                native,
                views: RefCell::new(Vec::new()),
            }),
        }
    }
    pub(crate) fn complete_encoded(
        &self,
        native: ffi::EncodedArray,
        recovered: Option<PcuExecutionFault>,
    ) -> MlxEncodedCompletion {
        MlxEncodedCompletion {
            output: self.wrap_encoded(native),
            recovered,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "Requires actual official UInt carriers and initialized unaligned ordinary RAM staging."]
    fn all_twenty_two_unaligned_logical_spans() {
        let runtime = crate::MlxRuntime::load_default().unwrap();
        let session = runtime.open_gpu(0).unwrap();
        for scalar in fusion_pcu::PcuScalarType::ALL
            .into_iter()
            .filter(|scalar| scalar.bit_width() >= 8)
        {
            let bytes = usize::from(scalar.bit_width()) / 8 * 7;
            let mut input = vec![0_u8; bytes + 4];
            let offset = (5 - input.as_ptr().addr() % 4) % 4;
            let logical = &mut input[offset..offset + bytes];
            for (index, byte) in logical.iter_mut().enumerate() {
                *byte = u8::try_from(index % 256)
                    .unwrap()
                    .wrapping_mul(37)
                    .wrapping_add(19);
            }
            assert_eq!(logical.as_ptr().addr() % 4, 1);
            let owner = session.upload_encoded_bytes(scalar, 7, logical).unwrap();
            let mut output = vec![0_u8; bytes];
            owner.read_bytes_into(&mut output).unwrap();
            assert_eq!(output, logical);
            assert_eq!(owner.scalar_type(), scalar);
            assert_eq!(owner.element_count(), 7);
            assert_eq!(owner.byte_len(), bytes);
        }
    }
}

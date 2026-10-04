//! A checked Shared-memory span; no fallible work remains at publication.
use core::marker::PhantomData;
use core::ptr::NonNull;
/// Only native checked Shared buffers can construct this capability. The source
/// and exclusive destination borrows retain both Objective-C owners through copy.
pub struct SharedPrefixCopy<'a> {
    source: NonNull<u8>,
    output: NonNull<u8>,
    bytes: usize,
    _owners: PhantomData<(&'a super::Buffer, &'a mut super::Buffer)>,
}
impl SharedPrefixCopy<'_> {
    #[cfg(target_os = "macos")]
    pub(in crate::ffi) const fn new(
        source: NonNull<u8>,
        output: NonNull<u8>,
        bytes: usize,
    ) -> Self {
        Self {
            source,
            output,
            bytes,
            _owners: PhantomData,
        }
    }
    pub const fn publish(self) {
        // SAFETY: originating session is terminal, both retained owners use
        // coherent CPU-visible Shared storage, source bytes are initialized,
        // extents were checked against logical and actual native lengths, and
        // output has an exclusive lease. Overlap-safe copy avoids an alias-based
        // nonoverlap assumption even though conversion payloads are fresh.
        unsafe {
            core::ptr::copy(self.source.as_ptr(), self.output.as_ptr(), self.bytes);
        }
    }
}

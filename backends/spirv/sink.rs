//! SPIR-V output sinks.

use super::error::PcuSpirvError;

/// Word sink used by SPIR-V lowering.
pub trait PcuSpirvSink {
    /// Appends one SPIR-V word.
    ///
    /// # Errors
    ///
    /// Returns an honest sink failure without partially hiding capacity exhaustion.
    fn push_word(&mut self, word: u32) -> Result<(), PcuSpirvError>;
}

/// Fixed-capacity no-alloc SPIR-V word sink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcuSpirvFixedSink<const WORDS: usize> {
    words: [u32; WORDS],
    len: usize,
}

impl<const WORDS: usize> PcuSpirvFixedSink<WORDS> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            words: [0; WORDS],
            len: 0,
        }
    }

    #[must_use]
    pub const fn as_slice(&self) -> &[u32] {
        self.words.split_at(self.len).0
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<const WORDS: usize> Default for PcuSpirvFixedSink<WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WORDS: usize> PcuSpirvSink for PcuSpirvFixedSink<WORDS> {
    fn push_word(&mut self, word: u32) -> Result<(), PcuSpirvError> {
        let Some(slot) = self.words.get_mut(self.len) else {
            return Err(PcuSpirvError::SinkFull);
        };
        *slot = word;
        self.len += 1;
        Ok(())
    }
}

//! Exact known compiler-template identities, independent of filesystems and allocation.
use sha2::{Digest, Sha256};
use crate::PcuSpirvError;

/// GLSL-produced arithmetic helper family. Low includes F16/BF16/F8 carriers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcuSpirvComposedTemplateFamily {
    F32,
    F64,
    Low,
    IntegerWide,
    IntegerNarrow,
}
/// Revision of the exact GLSL helper and frozen call rewriting metadata.
pub const PCU_COMPOSED_TEMPLATE_REVISION: &[u8] = b"PCU-COMPOSED-TEMPLATE-1-GLSL-REWRITE-1";

/// Borrowed external shader and all rewrite metadata. Validation requires the known digest;
/// structural validity alone does not establish numerical or resource contracts.
#[derive(Clone, Copy)]
pub struct PcuSpirvComposedTemplateData<'a> {
    pub family: PcuSpirvComposedTemplateFamily,
    pub words: &'a [u32],
    pub function_ids: &'a [u32],
    pub call_target_offsets: &'a [usize],
    pub call_argument_ids: &'a [u32],
    pub specialization_offsets: &'a [usize],
}
/// Sealed exact template, validated once during cold source loading.
#[derive(Clone, Copy)]
pub struct PcuSpirvComposedTemplate<'a>(PcuSpirvComposedTemplateData<'a>);
impl<'a> PcuSpirvComposedTemplate<'a> {
    /// Checks complete words and rewriting metadata against this compiler's exact known asset.
    /// # Errors
    /// Refuses any revision/family/words/metadata change before shader emission.
    pub fn validate(data: PcuSpirvComposedTemplateData<'a>) -> Result<Self, PcuSpirvError> {
        let mut digest = Sha256::new();
        digest.update(PCU_COMPOSED_TEMPLATE_REVISION);
        digest.update(
            match data.family {
                PcuSpirvComposedTemplateFamily::F32 => 0_u32,
                PcuSpirvComposedTemplateFamily::F64 => 1,
                PcuSpirvComposedTemplateFamily::Low => 2,
                PcuSpirvComposedTemplateFamily::IntegerWide => 3,
                PcuSpirvComposedTemplateFamily::IntegerNarrow => 4,
            }
            .to_le_bytes(),
        );
        hash_words(&mut digest, data.words);
        hash_words(&mut digest, data.function_ids);
        hash_offsets(&mut digest, data.call_target_offsets);
        hash_words(&mut digest, data.call_argument_ids);
        hash_offsets(&mut digest, data.specialization_offsets);
        let expected = match data.family {
            PcuSpirvComposedTemplateFamily::F32 => F32_IDENTITY,
            PcuSpirvComposedTemplateFamily::F64 => F64_IDENTITY,
            PcuSpirvComposedTemplateFamily::Low => LOW_IDENTITY,
            PcuSpirvComposedTemplateFamily::IntegerWide => INTEGER_WIDE_IDENTITY,
            PcuSpirvComposedTemplateFamily::IntegerNarrow => INTEGER_NARROW_IDENTITY,
        };
        if digest.finalize()[..] != expected {
            return Err(PcuSpirvError::InvalidKernelSignature);
        }
        Ok(Self(data))
    }
    #[must_use]
    pub const fn data(self) -> PcuSpirvComposedTemplateData<'a> {
        self.0
    }
}
fn hash_words(digest: &mut Sha256, words: &[u32]) {
    digest.update((words.len() as u64).to_le_bytes());
    for word in words {
        digest.update(word.to_le_bytes());
    }
}
fn hash_offsets(digest: &mut Sha256, offsets: &[usize]) {
    digest.update((offsets.len() as u64).to_le_bytes());
    for offset in offsets {
        digest.update((*offset as u64).to_le_bytes());
    }
}
const F32_IDENTITY: [u8; 32] = [
    0x2a, 0x0f, 0x67, 0x4c, 0x80, 0xb2, 0xea, 0xcd, 0xb5, 0x2e, 0x76, 0x2e, 0xee, 0xb5, 0x16, 0xf5,
    0xbc, 0x3a, 0x7a, 0xf5, 0x65, 0x51, 0xf3, 0x9b, 0x11, 0x8f, 0xde, 0xc2, 0x0a, 0x33, 0xf5, 0xf3,
];
const F64_IDENTITY: [u8; 32] = [
    0xef, 0x22, 0x6b, 0x3c, 0xdc, 0x77, 0xcc, 0x51, 0xd3, 0xbc, 0xea, 0xb5, 0x1f, 0xbd, 0xb6, 0xa8,
    0x6a, 0x68, 0xb1, 0x38, 0x6d, 0xd2, 0x2d, 0xed, 0xb7, 0xba, 0x6a, 0xec, 0x21, 0x51, 0x2d, 0xc7,
];
const LOW_IDENTITY: [u8; 32] = [
    0xb9, 0x51, 0x85, 0x84, 0x19, 0x08, 0x03, 0x74, 0x69, 0x10, 0x6e, 0x13, 0x4d, 0x80, 0xf7, 0xd4,
    0x19, 0x6d, 0xc1, 0xfa, 0xa1, 0xea, 0xa6, 0x58, 0xa6, 0x3f, 0xe7, 0xbe, 0xdf, 0x3a, 0xbd, 0x25,
];

const INTEGER_WIDE_IDENTITY: [u8; 32] = [
    0x92, 0x56, 0x3a, 0x05, 0x3b, 0x93, 0x59, 0xc4, 0xc1, 0x4f, 0xee, 0xc0, 0xa6, 0xf0, 0xae, 0x68,
    0x3b, 0x69, 0xf1, 0xba, 0x9c, 0x5a, 0x63, 0xdd, 0x81, 0x46, 0xf4, 0xe8, 0xa9, 0x6f, 0xc9, 0xd3,
];

const INTEGER_NARROW_IDENTITY: [u8; 32] = [
    0xec, 0xe2, 0xda, 0xa0, 0x0d, 0x7b, 0x03, 0x78, 0x54, 0x4f, 0xc2, 0xd0, 0x6d, 0x01, 0xa8, 0x30,
    0x5c, 0x3d, 0x45, 0xaf, 0x66, 0x4c, 0xd8, 0xce, 0xd1, 0x4b, 0x2e, 0x86, 0xc8, 0x44, 0xf7, 0xef,
];

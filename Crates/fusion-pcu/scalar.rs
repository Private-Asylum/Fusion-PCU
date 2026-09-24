//! Portable scalar identities and lossless canonical encodings.
//!
//! This contract describes supported Rust values, not a promise that a particular backend can
//! execute every operation on them or map host memory directly into a device address space.

use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuScalarType,
    PcuValueType,
};

mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for u32 {}
}

/// Scalar with a verified PCU identity and a lossless little-endian transfer encoding.
///
/// The trait is sealed until custom scalar representations have an explicit unsafe contract.
/// Backend admission still decides whether the type, operations, and physical layout are
/// supported. The host layout constants are not device ABI guarantees.
#[allow(private_bounds)] // A private supertrait prevents unchecked downstream implementations.
pub trait PcuScalar: Copy + sealed::Sealed {
    /// Semantic scalar identity in PCU IR.
    const TYPE: PcuScalarType;
    /// Size of the Rust value in host memory.
    const HOST_SIZE: usize;
    /// Alignment of the Rust value in host memory.
    const HOST_ALIGNMENT: usize;
    /// Size of the canonical little-endian encoding.
    const ENCODED_SIZE: usize;
    /// Fixed-size byte representation used for canonical transfer.
    type Encoded: AsRef<[u8]> + Copy;

    /// Encodes this value without changing its bit pattern.
    fn encode_le(self) -> Self::Encoded;

    /// Decodes an encoded value without changing its bit pattern.
    fn decode_le(bytes: Self::Encoded) -> Self;
}

impl PcuScalar for f32 {
    const TYPE: PcuScalarType = PcuScalarType::F32;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 4;
    type Encoded = [u8; 4];

    fn encode_le(self) -> Self::Encoded {
        self.to_bits().to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_bits(u32::from_le_bytes(bytes))
    }
}

impl PcuScalar for u32 {
    const TYPE: PcuScalarType = PcuScalarType::U32;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 4;
    type Encoded = [u8; 4];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl<'a> PcuBinding<'a> {
    /// Declares a scalar resource with its IR element type derived from `T`.
    ///
    /// This does not establish device placement or a zero-copy host representation.
    #[must_use]
    pub const fn scalar<T: PcuScalar>(
        name: Option<&'a str>,
        set: u32,
        binding: u32,
        storage: PcuBindingStorageClass,
        access: PcuBindingAccess,
    ) -> Self {
        Self::value(
            name,
            set,
            binding,
            storage,
            access,
            PcuValueType::Scalar(T::TYPE),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::PcuScalar;
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingStorageClass,
        PcuScalarType,
        PcuValueType,
    };

    #[test]
    fn scalar_encodings_preserve_integer_and_float_bit_patterns() {
        let integer = 0x1234_56ab_u32;
        assert_eq!(integer.encode_le(), [0xab, 0x56, 0x34, 0x12]);
        assert_eq!(u32::decode_le(integer.encode_le()), integer);

        let nan = f32::from_bits(0x7fc0_1234);
        assert_eq!(f32::decode_le(nan.encode_le()).to_bits(), nan.to_bits());
        assert_eq!(<f32 as PcuScalar>::TYPE, PcuScalarType::F32);
        assert_eq!(<u32 as PcuScalar>::TYPE, PcuScalarType::U32);
        assert_eq!(<f32 as PcuScalar>::ENCODED_SIZE, 4);
        assert_eq!(
            <u32 as PcuScalar>::HOST_ALIGNMENT,
            core::mem::align_of::<u32>()
        );

        let scalar = PcuBinding::scalar::<u32>(
            Some("values"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        );
        assert_eq!(scalar.value_type(), Some(PcuValueType::u32()));
    }
}

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
    impl Sealed for f64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
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

impl PcuScalar for f64 {
    const TYPE: PcuScalarType = PcuScalarType::F64;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 8;
    type Encoded = [u8; 8];

    fn encode_le(self) -> Self::Encoded {
        self.to_bits().to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_bits(u64::from_le_bytes(bytes))
    }
}

impl PcuScalar for u8 {
    const TYPE: PcuScalarType = PcuScalarType::U8;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 1;
    type Encoded = [Self; 1];

    fn encode_le(self) -> Self::Encoded {
        [self]
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        bytes[0]
    }
}

impl PcuScalar for u16 {
    const TYPE: PcuScalarType = PcuScalarType::U16;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 2;
    type Encoded = [u8; 2];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
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

impl PcuScalar for u64 {
    const TYPE: PcuScalarType = PcuScalarType::U64;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 8;
    type Encoded = [u8; 8];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }
    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl PcuScalar for i8 {
    const TYPE: PcuScalarType = PcuScalarType::I8;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 1;
    type Encoded = [u8; 1];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl PcuScalar for i16 {
    const TYPE: PcuScalarType = PcuScalarType::I16;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 2;
    type Encoded = [u8; 2];

    fn encode_le(self) -> Self::Encoded {
        self.to_le_bytes()
    }

    fn decode_le(bytes: Self::Encoded) -> Self {
        Self::from_le_bytes(bytes)
    }
}

impl PcuScalar for i32 {
    const TYPE: PcuScalarType = PcuScalarType::I32;
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

impl PcuScalar for i64 {
    const TYPE: PcuScalarType = PcuScalarType::I64;
    const HOST_SIZE: usize = core::mem::size_of::<Self>();
    const HOST_ALIGNMENT: usize = core::mem::align_of::<Self>();
    const ENCODED_SIZE: usize = 8;
    type Encoded = [u8; 8];

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
    fn i8_scalar_encoding_preserves_twos_complement_bits() {
        let value = i8::MIN + 37;
        assert_eq!(i8::decode_le(value.encode_le()), value);
        assert_eq!(<i8 as PcuScalar>::TYPE, PcuScalarType::I8);
        assert_eq!(<i8 as PcuScalar>::ENCODED_SIZE, 1);
        assert_eq!(<i8 as PcuScalar>::HOST_SIZE, 1);
    }

    #[test]
    fn scalar_encodings_preserve_integer_and_float_bit_patterns() {
        let integer = 0x1234_56ab_u32;
        assert_eq!(integer.encode_le(), [0xab, 0x56, 0x34, 0x12]);
        assert_eq!(u32::decode_le(integer.encode_le()), integer);
        let wide = 0x0123_4567_89ab_cdef_u64;
        assert_eq!(
            wide.encode_le(),
            [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]
        );
        assert_eq!(u64::decode_le(wide.encode_le()), wide);
        let signed32 = i32::MIN + 0x1234;
        assert_eq!(i32::decode_le(signed32.encode_le()), signed32);
        assert_eq!(<i32 as PcuScalar>::TYPE, PcuScalarType::I32);
        let signed = i64::MIN + 0x0123_4567;
        assert_eq!(i64::decode_le(signed.encode_le()), signed);
        assert_eq!(<i64 as PcuScalar>::TYPE, PcuScalarType::I64);
        assert_eq!(<u64 as PcuScalar>::TYPE, PcuScalarType::U64);

        let nan = f32::from_bits(0x7fc0_1234);
        assert_eq!(f32::decode_le(nan.encode_le()).to_bits(), nan.to_bits());
        assert_eq!(<f32 as PcuScalar>::TYPE, PcuScalarType::F32);
        let wide_nan = f64::from_bits(0x7ff8_0000_0000_1234);
        assert_eq!(
            f64::decode_le(wide_nan.encode_le()).to_bits(),
            wide_nan.to_bits()
        );
        assert_eq!(<f64 as PcuScalar>::TYPE, PcuScalarType::F64);
        assert_eq!(<f64 as PcuScalar>::ENCODED_SIZE, 8);
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

    #[test]
    fn narrow_scalar_encodings_preserve_bits_and_type_identity() {
        let signed16 = i16::MIN + 0x1234;
        assert_eq!(i16::decode_le(signed16.encode_le()), signed16);
        assert_eq!(<i16 as PcuScalar>::TYPE, PcuScalarType::I16);
        let byte = 0xb2_u8;
        assert_eq!(u8::decode_le(byte.encode_le()), byte);
        assert_eq!(<u8 as PcuScalar>::TYPE, PcuScalarType::U8);
        assert_eq!(<u8 as PcuScalar>::ENCODED_SIZE, 1);
        let narrow = 0xa1b2_u16;
        assert_eq!(u16::decode_le(narrow.encode_le()), narrow);
        assert_eq!(<u16 as PcuScalar>::TYPE, PcuScalarType::U16);
        assert_eq!(<u16 as PcuScalar>::ENCODED_SIZE, 2);
    }
}

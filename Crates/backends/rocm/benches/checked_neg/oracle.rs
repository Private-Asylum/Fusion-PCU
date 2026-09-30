//! Exact encoding oracles do not evaluate floating negation or share device arithmetic helpers.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuValueType,
    PcuValueTypeCaps,
};

pub trait Scalar: PcuScalar + std::fmt::Debug {
    const NAME: &'static str;
    const VALUE_TYPE: PcuValueType;
    const CAPS: PcuValueTypeCaps;
    const SIGN: u64;
    fn sample(index: usize, phase: u64) -> Self;
    fn invalid() -> Self;
    fn tiny() -> Self;
    fn encoding(self) -> u64;
    fn write(self, bytes: &mut [u8]);
    fn read(bytes: &[u8]) -> u64;
    fn source<const N: usize>(input: &[Self], output: &mut [Self])
    -> Result<(), PcuExecutionError>;
    fn reject_subnormal<const N: usize>(
        input: &[Self],
        output: &mut [Self],
    ) -> Result<(), PcuExecutionError>;
    fn native<const N: usize>(
        backend: &fusion_pcu_rocm::RocmOwnedDispatchBackend,
    ) -> super::native::Native;
}

macro_rules! scalar {
    ($float:ty, $uint:ty, $name:literal, $type:expr, $caps:expr, $sign:expr, $finite_mask:expr, $source:ident, $reject:ident, $bindings:ident, $ir:ident) => {
        impl Scalar for $float {
            const NAME: &'static str = $name;
            const VALUE_TYPE: PcuValueType = $type;
            const CAPS: PcuValueTypeCaps = $caps;
            const SIGN: u64 = $sign;

            fn sample(index: usize, phase: u64) -> Self {
                let value = u64::try_from(index)
                    .unwrap()
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(phase.wrapping_mul(0xbf58_476d_1ce4_e5b9));
                #[allow(clippy::cast_possible_truncation)]
                // Deliberately choose the format's encoding width.
                let encoding = value as $uint;
                Self::from_bits(encoding & $finite_mask)
            }
            fn invalid() -> Self {
                Self::INFINITY
            }
            fn tiny() -> Self {
                Self::from_bits(1)
            }
            #[allow(clippy::useless_conversion)] // One macro handles u32 and u64 encodings.
            fn encoding(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn write(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_ne_bytes());
            }
            #[allow(clippy::useless_conversion)] // The width-dependent encoding converts to the oracle's common type.
            fn read(bytes: &[u8]) -> u64 {
                u64::from(<$uint>::from_ne_bytes(bytes.try_into().unwrap()))
            }
            fn source<const N: usize>(
                input: &[Self],
                output: &mut [Self],
            ) -> Result<(), PcuExecutionError> {
                super::source::$source::<N>(input, output)
            }
            fn reject_subnormal<const N: usize>(
                input: &[Self],
                output: &mut [Self],
            ) -> Result<(), PcuExecutionError> {
                super::source::$reject::<N>(input, output)
            }
            fn native<const N: usize>(
                backend: &fusion_pcu_rocm::RocmOwnedDispatchBackend,
            ) -> super::native::Native {
                let bindings = super::source::$bindings();
                let builder = super::source::$ir::<N>(&bindings).unwrap();
                builder.with_ir(|ir| super::native::Native::new(backend, ir, N * size_of::<Self>()))
            }
        }
    };
}

scalar!(
    f32,
    u32,
    "f32",
    PcuValueType::f32(),
    PcuValueTypeCaps::FLOAT32,
    0x8000_0000,
    0xff7f_ffff,
    negate_f32,
    negate_f32_reject_subnormal,
    negate_f32_bindings,
    negate_f32_ir
);
scalar!(
    f64,
    u64,
    "f64",
    PcuValueType::f64(),
    PcuValueTypeCaps::FLOAT64,
    0x8000_0000_0000_0000,
    0xffef_ffff_ffff_ffff,
    negate_f64,
    negate_f64_reject_subnormal,
    negate_f64_bindings,
    negate_f64_ir
);

pub fn input<T: Scalar>(count: usize, phase: u64) -> Vec<T> {
    (0..count).map(|index| T::sample(index, phase)).collect()
}

#[allow(clippy::chunks_exact_to_as_chunks)] // Stable const arguments cannot use generic size_of::<T>().
pub fn bytes<T: Scalar>(values: &[T]) -> Vec<u8> {
    let mut bytes = vec![0; size_of_val(values)];
    for (value, chunk) in values.iter().zip(bytes.chunks_exact_mut(size_of::<T>())) {
        value.write(chunk);
    }
    bytes
}

pub fn verify<T: Scalar>(input: &[T], output: &[T]) {
    assert_eq!(input.len(), output.len());
    for (index, (input, output)) in input.iter().zip(output).enumerate() {
        assert_eq!(
            output.encoding(),
            input.encoding() ^ T::SIGN,
            "{} Neg encoding at {index}",
            T::NAME
        );
    }
}

#[allow(clippy::chunks_exact_to_as_chunks)] // Stable const arguments cannot use generic size_of::<T>().
pub fn verify_bytes<T: Scalar>(input: &[T], output: &[u8]) {
    assert_eq!(size_of_val(input), output.len());
    for (index, (input, output)) in input
        .iter()
        .zip(output.chunks_exact(size_of::<T>()))
        .enumerate()
    {
        assert_eq!(
            T::read(output),
            input.encoding() ^ T::SIGN,
            "{} native Neg encoding at {index}",
            T::NAME
        );
    }
}

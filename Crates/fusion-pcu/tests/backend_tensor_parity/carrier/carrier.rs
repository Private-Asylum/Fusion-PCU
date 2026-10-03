//! Raw owned transport preserves every carrier's full representation and Rust lifetime.

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuBf16Bits,
    PcuCompoundArithmeticPolicy,
    PcuExecutionError,
    PcuF16Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuI256,
    PcuI512,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuScalar,
    PcuTensor,
    PcuU256,
    PcuU512,
};
use super::contract::POLICY_LOCK;

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

trait Sample: PcuScalar {
    fn sample(seed: u8) -> Self;
}

macro_rules! samples {
    ($($ty:ty => $width:literal;)+) => {$(
        impl Sample for $ty {
            fn sample(seed: u8) -> Self {
                let mut bytes = [0_u8; $width];
                for (index, byte) in bytes.iter_mut().enumerate() {
                    *byte = match seed {
                        0 => u8::MAX, // Includes NaN payloads in floating carriers.
                        1 => 0,
                        2 => if index + 1 == $width { 0x80 } else { 0 },
                        _ => seed.wrapping_mul(37).wrapping_add(u8::try_from(index).unwrap().wrapping_mul(19)),
                    };
                }
                Self::decode_le(bytes)
            }
        }
    )+};
}
samples! {
    u8 => 1; i8 => 1; u16 => 2; i16 => 2;
    u32 => 4; i32 => 4; u64 => 8; i64 => 8;
    u128 => 16; i128 => 16; PcuU256 => 32; PcuI256 => 32;
    PcuU512 => 64; PcuI512 => 64;
    PcuF16Bits => 2; PcuBf16Bits => 2;
    PcuF8E4M3FnBits => 1; PcuF8E5M2Bits => 1;
    f32 => 4; f64 => 8; PcuF128Bits => 16; PcuF256Bits => 32;
}

fn check<T: Sample>(owner: &PcuTensor<T>, expected: &[T; 7]) {
    assert_eq!(owner.shape(), &[7]);
    let sentinel = T::sample(91);
    let mut stack = [sentinel; 9];
    owner.read_into(&mut stack).unwrap();
    for (actual, expected) in stack[..7].iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    for tail in &stack[7..] {
        assert_eq!(tail.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
    let mut short = [sentinel; 6];
    assert!(owner.read_into(&mut short).is_err());
    for value in short {
        assert_eq!(value.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}

fn format<T: Sample>() {
    for phase in [0_u8, 23, 71] {
        let mut input = [0_u8, 1, 2, 3, 5, 7, 11].map(|seed| T::sample(seed.wrapping_add(phase)));
        let expected = input;
        let first = retain(&input).unwrap();
        input.fill(T::sample(99)); // The escaped owner must not alias mutable host input.
        check(&first, &expected);
        let second = retain(&first).unwrap();
        let sibling = retain(&first).unwrap();
        drop(first);
        check(&second, &expected);
        let moved = consume(second).unwrap();
        check(&moved, &expected);
        global::clear_thread_cache().unwrap();
        let after_clear = retain(&sibling).unwrap();
        drop(sibling);
        check(&after_clear, &expected);
        check(&moved, &expected);
        drop(moved);
        check(&after_clear, &expected);
    }
}

/// All22 representations are required, including NaN payloads and full high limbs.
pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    global::configure(global::PcuExecutionPolicy {
                        backend,
                        numerical_mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            ..Default::default()
                        },
                        float_underflow,
                        ..Default::default()
                    })
                    .unwrap();
                    format::<u8>();
                    format::<i8>();
                    format::<u16>();
                    format::<i16>();
                    format::<u32>();
                    format::<i32>();
                    format::<u64>();
                    format::<i64>();
                    format::<u128>();
                    format::<i128>();
                    format::<PcuU256>();
                    format::<PcuI256>();
                    format::<PcuU512>();
                    format::<PcuI512>();
                    format::<PcuF16Bits>();
                    format::<PcuBf16Bits>();
                    format::<PcuF8E4M3FnBits>();
                    format::<PcuF8E5M2Bits>();
                    format::<f32>();
                    format::<f64>();
                    format::<PcuF128Bits>();
                    format::<PcuF256Bits>();
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[path = "prefix/prefix.rs"]
pub mod prefix;

//! Exact host/resident transport for every sealed scalar type.
#![cfg(feature = "tensor")]

use fusion_pcu::pcu;
#[cfg(feature = "rocm")]
use fusion_pcu::global;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
    PcuBf16Bits,
    PcuF16Bits,
};

#[pcu]
fn copy<T: PcuScalar, const N: usize>(input: &[T; N]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn copy_through_helper<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(copy::<T, N>(input)?)
}

#[pcu]
fn consume_copy<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[cfg(feature = "rocm")]
fn encoded<T: PcuScalar>(values: &[T]) -> std::vec::Vec<u8> {
    let mut bytes = std::vec::Vec::with_capacity(values.len() * T::ENCODED_SIZE);
    for value in values {
        bytes.extend_from_slice(value.encode_le().as_ref());
    }
    bytes
}

#[cfg(feature = "rocm")]
fn exercise_transport<T: PcuScalar, const N: usize>(first: &[T; N], changed: &[T; N]) {
    let fresh = copy(first).expect("input-only fresh output");
    let helper_first = copy_through_helper(first).expect("helper input-only output");
    let helper_changed = copy_through_helper(changed).expect("warm helper input-only output");
    let consumed_input = copy(first).expect("consuming input source");
    let consumed = consume_copy(consumed_input).expect("consuming input-only output");

    // Cache eviction must release prepared state while escaped outputs retain their roots/storage.
    global::clear_thread_cache().unwrap();

    let first_bytes = encoded(first);
    let changed_bytes = encoded(changed);
    let mut observed = [first[0]; N];
    fresh.read_into(&mut observed).unwrap();
    assert_eq!(encoded(&observed), first_bytes);
    helper_first.read_into(&mut observed).unwrap();
    assert_eq!(encoded(&observed), first_bytes);
    helper_changed.read_into(&mut observed).unwrap();
    assert_eq!(encoded(&observed), changed_bytes);
    consumed.read_into(&mut observed).unwrap();
    assert_eq!(encoded(&observed), first_bytes);
}

#[test]
#[cfg(not(feature = "rocm"))]
fn every_scalar_transport_profile_compiles_without_cpu_fallback() {
    let _: fn(&[i8; 4]) -> Result<PcuTensor<i8>, PcuExecutionError> = copy::<i8, 4>;
    let _: fn(&[u8; 4]) -> Result<PcuTensor<u8>, PcuExecutionError> = copy::<u8, 4>;
    let _: fn(&[i16; 4]) -> Result<PcuTensor<i16>, PcuExecutionError> = copy::<i16, 4>;
    let _: fn(&[u16; 4]) -> Result<PcuTensor<u16>, PcuExecutionError> = copy::<u16, 4>;
    let _: fn(&[i32; 4]) -> Result<PcuTensor<i32>, PcuExecutionError> = copy::<i32, 4>;
    let _: fn(&[u32; 4]) -> Result<PcuTensor<u32>, PcuExecutionError> = copy::<u32, 4>;
    let _: fn(&[i64; 4]) -> Result<PcuTensor<i64>, PcuExecutionError> = copy::<i64, 4>;
    let _: fn(&[u64; 4]) -> Result<PcuTensor<u64>, PcuExecutionError> = copy::<u64, 4>;
    let _: fn(&[f32; 4]) -> Result<PcuTensor<f32>, PcuExecutionError> = copy::<f32, 4>;
    let _: fn(&[f64; 4]) -> Result<PcuTensor<f64>, PcuExecutionError> = copy::<f64, 4>;
    let _: fn(&[PcuF16Bits; 4]) -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> =
        copy::<PcuF16Bits, 4>;
    let _: fn(&[PcuBf16Bits; 4]) -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> =
        copy::<PcuBf16Bits, 4>;
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware; verifies byte-exact transport for all PcuScalar types"]
fn all_scalar_types_round_trip_through_fresh_helper_and_consuming_paths() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    exercise_transport(&[i8::MIN, i8::MAX, -1, 0], &[i8::MAX, i8::MIN, 0, 1]);
    exercise_transport(&[0_u8, u8::MAX, 0x80, 0x5a], &[u8::MAX, 0, 1, 0xa5]);
    exercise_transport(&[i16::MIN, i16::MAX, -1, 0], &[i16::MAX, i16::MIN, 0, 1]);
    exercise_transport(
        &[0_u16, u16::MAX, 0x8000, 0x5aa5],
        &[u16::MAX, 0, 1, 0xa55a],
    );
    exercise_transport(&[i32::MIN, i32::MAX, -1, 0], &[i32::MAX, i32::MIN, 0, 1]);
    exercise_transport(
        &[0_u32, u32::MAX, 0x8000_0001, 0xdead_beef],
        &[u32::MAX, 0, 1, 0x0123_4567],
    );
    exercise_transport(&[i64::MIN, i64::MAX, -1, 0], &[i64::MAX, i64::MIN, 0, 1]);
    exercise_transport(
        &[
            0_u64,
            u64::MAX,
            0x8000_0000_0000_0001,
            0xdead_beef_cafe_babe,
        ],
        &[u64::MAX, 0, 1, 0x0123_4567_89ab_cdef],
    );
    exercise_transport(
        &[
            f32::from_bits(0x8000_0000),
            f32::from_bits(0x7fc1_2345),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
        &[
            f32::from_bits(0x0000_0000),
            f32::from_bits(0xffa5_4321),
            f32::MIN_POSITIVE,
            -1.0,
        ],
    );
    exercise_transport(
        &[
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x7ff8_1234_5678_9abc),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
        &[
            f64::from_bits(0x0000_0000_0000_0000),
            f64::from_bits(0xfff8_cafe_dead_beef),
            f64::MIN_POSITIVE,
            -1.0,
        ],
    );
    exercise_transport(
        &[
            PcuF16Bits::from_bits(0x8000),
            PcuF16Bits::from_bits(0x7c01),
            PcuF16Bits::from_bits(0xfc00),
            PcuF16Bits::from_bits(0x0001),
        ],
        &[
            PcuF16Bits::from_bits(0x0000),
            PcuF16Bits::from_bits(0x7d55),
            PcuF16Bits::from_bits(0x3c00),
            PcuF16Bits::from_bits(0x8001),
        ],
    );
    exercise_transport(
        &[
            PcuBf16Bits::from_bits(0x8000),
            PcuBf16Bits::from_bits(0x7f81),
            PcuBf16Bits::from_bits(0xff80),
            PcuBf16Bits::from_bits(0x0001),
        ],
        &[
            PcuBf16Bits::from_bits(0x0000),
            PcuBf16Bits::from_bits(0x7f93),
            PcuBf16Bits::from_bits(0x3f80),
            PcuBf16Bits::from_bits(0x8001),
        ],
    );
}

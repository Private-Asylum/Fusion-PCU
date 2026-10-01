#[rustfmt::skip]
use super::{
    as_f32,
    as_f32_mut,
    MlxError,
    PcuScalar,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBf16Bits,
    PcuF16Bits,
};

fn rejected<T: PcuScalar + PartialEq + std::fmt::Debug>(value: T) {
    let mut storage = [value; 2];
    assert_eq!(as_f32(&storage), Err(MlxError::UnsupportedScalar(T::TYPE)));
    assert_eq!(
        as_f32_mut(&mut storage),
        Err(MlxError::UnsupportedScalar(T::TYPE))
    );
    assert_eq!(storage, [value; 2]);
}

#[test]
fn equal_width_and_other_sealed_scalars_are_not_rust_f32() {
    rejected(17_i32);
    rejected(17_u32);
    rejected(17_f64);
    rejected(17_i8);
    rejected(17_i16);
    rejected(17_i64);
    rejected(17_u8);
    rejected(17_u16);
    rejected(17_u64);
    rejected(PcuF16Bits::from_bits(17));
    rejected(PcuBf16Bits::from_bits(17));
}

#[test]
fn exact_f32_borrows_preserve_pointer_bits_extent_and_exclusive_writes() {
    let words = [0_u32, 0x8000_0000, 1, 0x8000_0001, 0x7fc1_2345, 0xff80_0000];
    let mut storage = words.map(f32::from_bits);
    let view = as_f32(&storage).unwrap();
    assert_eq!(view.as_ptr(), storage.as_ptr());
    assert_eq!(view.len(), storage.len());
    assert_eq!(
        view.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
        words
    );
    let pointer = storage.as_mut_ptr();
    let view = as_f32_mut(&mut storage).unwrap();
    assert_eq!(view.as_mut_ptr(), pointer);
    view[0] = f32::from_bits(0x8000_0000);
    assert_eq!(storage[0].to_bits(), 0x8000_0000);
    assert_eq!(as_f32::<f32>(&[]).unwrap(), []);
}

//! Independent raw representation fixture/control; no arithmetic implementation is reused.
use pcu_facade::PcuScalar;

pub fn sample<T: PcuScalar>(bank: u8, lane: usize) -> T {
    let mut value = core::mem::MaybeUninit::<T>::uninit();
    // SAFETY: The sealed host carriers selected by this fixture are padding-free and admit all
    // bit patterns. Initialize exactly size_of::<T>() bytes before constructing the aligned T.
    unsafe {
        let bytes = core::slice::from_raw_parts_mut(
            value.as_mut_ptr().cast::<u8>(),
            core::mem::size_of::<T>(),
        );
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = bank
                .wrapping_add(u8::try_from(lane % 256).unwrap())
                .wrapping_add(u8::try_from(index).unwrap().wrapping_mul(17));
        }
        value.assume_init()
    }
}
pub fn equal<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (lane, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.encode_le().as_ref(),
            expected.encode_le().as_ref(),
            "{:?} lane {lane}",
            T::TYPE
        );
    }
}
pub fn native<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    assert!(input.len() >= N && stage.len() >= N && output.len() >= N);
    // This ordinary Rust control owns a loaded value before overwriting its source, independently
    // of the provider's scratch/register indexing and source macro lowering.
    for lane in 0..N {
        stage[lane] = input[lane];
        let saved = stage[lane];
        stage[lane] = *seed;
        output[lane] = saved;
    }
}

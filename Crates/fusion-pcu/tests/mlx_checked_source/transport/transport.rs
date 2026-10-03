//! Native-required transport: Rust exclusive borrows provide immutable old snapshots.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[path = "fixtures/fixtures.rs"]
mod fixtures;
#[path = "source/source.rs"]
mod source;

const N: usize = 7;

const fn sample<T: PcuScalar>(byte: u8) -> T {
    let mut value = core::mem::MaybeUninit::<T>::uninit();
    // SAFETY: sealed PcuScalar carriers admit every bit pattern and have no
    // padding. Fill their entire aligned host representation before reading T.
    unsafe {
        value
            .as_mut_ptr()
            .cast::<u8>()
            .write_bytes(byte, core::mem::size_of::<T>());
        value.assume_init()
    }
}

fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.encode_le().as_ref(),
            expected.encode_le().as_ref(),
            "{index}: {:?}",
            T::TYPE
        );
    }
}

fn read<T: PcuScalar, const K: usize>(owner: &PcuTensor<T>, expected: &[T; K]) {
    let sentinel = sample::<T>(0xA5);
    let mut stack = [sentinel; K];
    owner.read_into(&mut stack).unwrap();
    bits(&stack, expected);
}

pub fn verify() {
    let _guard = super::POLICY_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    macro_rules! cases {
        ($($ty:ty),+ $(,)?) => { $(fixtures::verify::<$ty>();)+ };
    }
    cases!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        f32,
        f64,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        PcuF128Bits,
        PcuF256Bits,
        PcuU256,
        PcuI256,
        PcuU512,
        PcuI512
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

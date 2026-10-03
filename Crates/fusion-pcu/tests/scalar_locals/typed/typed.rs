//! Explicit annotations carry no device lifetime or numerical escape.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuRangePolicy,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};

#[pcu(invocations = N)]
fn typed_direct<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original: T = input[id];
    let mut value: T = original;
    let saved: _ = value;
    value = *seed;
    stage[id] = value;
    output[id] = saved;
}

#[pcu(invocations = 3)]
fn typed_grid<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let original: T = input[id];
        let mut value: T = original;
        let saved: _ = value;
        value = *seed;
        stage[id] = value;
        output[id] = saved;
        id += stride;
    }
}

const fn sample<T: PcuScalar>(byte: u8) -> T {
    let mut value = core::mem::MaybeUninit::<T>::uninit();
    // SAFETY: PcuScalar is sealed, padding-free and admits every bit pattern.
    // Initialize exactly the whole aligned carrier without forming a byte view
    // of uninitialized storage. Annotations must preserve all these raw bits.
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
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}
fn verify<T: PcuScalar>() {
    const N: usize = 47;
    let sentinel = sample::<T>(0xA5);
    let mut stage = [sentinel; N + 3];
    let mut output = [sentinel; N + 5];
    for phase in [0_u8, 85, 255] {
        let input: [T; N + 7] = core::array::from_fn(|index| {
            sample::<T>(phase.wrapping_add(u8::try_from(index).unwrap()))
        });
        let seed = sample::<T>(phase.wrapping_add(19));
        typed_direct::<T, N>(&input, &seed, &mut stage, &mut output).unwrap();
        bits(&stage[..N], &[seed; N]);
        bits(&output[..N], &input[..N]);
        bits(&stage[N..], &[sentinel; 3]);
        bits(&output[N..], &[sentinel; 5]);
        typed_grid::<T, N>(&input, &seed, &mut stage, &mut output).unwrap();
        bits(&stage[..N], &[seed; N]);
        bits(&output[..N], &input[..N]);
        bits(&stage[N..], &[sentinel; 3]);
        bits(&output[N..], &[sentinel; 5]);
        let before_stage = stage;
        let before_output = output;
        assert!(typed_direct::<T, N>(&input, &seed, &mut stage, &mut output[..N - 1]).is_err());
        bits(&stage, &before_stage);
        bits(&output, &before_output);
        typed_direct::<T, N>(&input, &seed, &mut stage, &mut output).unwrap();
        bits(&output[..N], &input[..N]);
    }
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_twenty_two_typed_carriers_preserve_saved_values_and_host_atomicity() {
    let _guard = POLICY_LOCK.lock().unwrap();
    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
        configure(PcuExecutionPolicy {
            backend: PcuBackendChoice::Cpu,
            range_policy,
            ..Default::default()
        })
        .unwrap();
        clear_thread_cache().unwrap();
        macro_rules! types { ($($ty:ty),+) => { $(verify::<$ty>();)+ }; }
        types!(
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
            PcuU256,
            PcuI256,
            PcuU512,
            PcuI512,
            PcuF16Bits,
            PcuBf16Bits,
            PcuF8E4M3FnBits,
            PcuF8E5M2Bits,
            f32,
            f64,
            PcuF128Bits,
            PcuF256Bits
        );
    }
    clear_thread_cache().unwrap();
    use_defaults().unwrap();
}

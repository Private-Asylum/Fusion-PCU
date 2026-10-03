//! Identity transports all sealed carriers and raw payloads; it does not perform arithmetic.

#[path = "broadcast/broadcast.rs"]
pub mod broadcast;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuBf16Bits,
    PcuF16Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuI256,
    PcuI512,
    PcuNumericalMode,
    PcuScalar,
    PcuU256,
    PcuU512,
};

#[rustfmt::skip]
use super::support::{
    bits,
    POLICY_LOCK,
};

#[pcu(invocations = N)]
fn direct<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[pcu(invocations = 17)]
fn grid<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}

trait Sample: PcuScalar {
    fn pattern(seed: u8) -> Self;
}

macro_rules! samples {
    ($($ty:ty),+ $(,)?) => {$(
        impl Sample for $ty {
            fn pattern(seed: u8) -> Self {
                // Every byte, including all high limbs, changes between replays. Unsigned
                // wrapping here deliberately creates arbitrary transport bits, not PCU math.
                Self::decode_le(std::array::from_fn(|index| {
                    seed.wrapping_add(u8::try_from(index).unwrap().wrapping_mul(37))
                }))
            }
        }
    )+};
}

samples!(
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
    PcuF256Bits,
);

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}

fn format<T: Sample>() {
    for seed in [0_u8, 73, 191] {
        let input: [T; 65] = std::array::from_fn(|index| {
            T::pattern(seed.wrapping_add(u8::try_from(index).unwrap()))
        });
        let sentinel = T::pattern(251);
        let mut output = [sentinel; 68];
        for execute in [direct::<T, 65>, grid::<T, 65>] {
            execute(&input, &mut output).unwrap();
            bits(&output[..65], &input);
            bits(&output[65..], &[sentinel; 3]);
            output.fill(sentinel);
        }
    }
}

fn all_formats() {
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

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        global::configure(global::PcuExecutionPolicy {
            backend,
            numerical_mode,
            score_invocation: Some(score),
            ..Default::default()
        })
        .unwrap();
        global::clear_thread_cache().unwrap();
        all_formats();
        let cold = SCORES.load(Ordering::Relaxed);
        all_formats();
        assert_eq!(SCORES.load(Ordering::Relaxed), cold);
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

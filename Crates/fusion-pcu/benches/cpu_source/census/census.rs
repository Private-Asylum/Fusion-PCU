//! Separately compile-gated work census; no observer in ordinary benchmark builds.
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
use fusion_pcu::global;
use super::native::Error;
use super::setup;

#[path = "allocator/allocator.rs"]
mod allocator;
pub use allocator::CountingAllocator;
use allocator::Counts;

static SCORES: AtomicUsize = AtomicUsize::new(0);

pub fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

pub fn helper_integer<const N: usize>(
    route: &str,
    mut execute: impl FnMut(&[u64; N], &u64, &mut [u64]) -> Result<(), Error>,
) {
    let mut input = setup::array::<u64, N>(1);
    let mut output = vec![17; N + 2];
    execute(&input, &1, &mut output).unwrap();
    let scores = SCORES.load(Ordering::Relaxed);
    let mut total = Counts::default();
    for phase in 0..64_u64 {
        input[0] = 1 + phase;
        let (result, counts) = allocator::observe(|| execute(&input, &1, &mut output));
        result.unwrap();
        total.allocations += counts.allocations;
        total.reallocations += counts.reallocations;
        total.frees += counts.frees;
        total.requested_bytes += counts.requested_bytes;
        for (index, &value) in input.iter().enumerate() {
            assert_eq!(output[index], value * value);
        }
        assert_eq!(output[N..], [17; 2]);
    }
    assert_eq!(
        total,
        Counts::default(),
        "{route}: warm caller Rust allocator work"
    );
    assert_eq!(
        SCORES.load(Ordering::Relaxed),
        scores,
        "{route}: warm rescoring"
    );
    println!(
        "cpu_source census/integer_helper/{N}/{route}:64 unique calls alloc={} realloc={} free={} bytes={} rescore=0",
        total.allocations, total.reallocations, total.frees, total.requested_bytes
    );
}

pub fn composed<const N: usize, const ORDERED: bool>(
    route: &str,
    mut execute: impl FnMut(&[f32; N], &mut [f32], &mut [f32]) -> Result<(), Error>,
) {
    let mut input = setup::array::<f32, N>(1.0);
    let mut stage = vec![16.0; N + 2];
    let mut output = vec![16.0; N + 2];
    execute(&input, &mut stage, &mut output).unwrap();
    let scores = SCORES.load(Ordering::Relaxed);
    let mut total = Counts::default();
    for phase in 0..64_u8 {
        // All64 calls have a different exactly represented input. Set up and
        // independently verify each full output outside its allocation scope.
        input[0] = 1.0 + f32::from(phase) / 64.0;
        let (result, counts) = allocator::observe(|| execute(&input, &mut stage, &mut output));
        result.unwrap();
        total.allocations += counts.allocations;
        total.reallocations += counts.reallocations;
        total.frees += counts.frees;
        total.requested_bytes += counts.requested_bytes;
        for (index, &value) in input.iter().enumerate() {
            // These dyadic products are exact in F32. This is an independent
            // value-domain oracle, not a general native checked-float claim.
            assert_eq!(output[index].to_bits(), ((value + value) * value).to_bits());
            assert_eq!(
                stage[index].to_bits(),
                if ORDERED {
                    (value + value).to_bits()
                } else {
                    16.0_f32.to_bits()
                }
            );
        }
        assert_eq!(output[N..], [16.0; 2]);
        assert_eq!(stage[N..], [16.0; 2]);
    }
    assert_eq!(
        total,
        Counts::default(),
        "{route}: warm caller Rust allocator work"
    );
    assert_eq!(
        SCORES.load(Ordering::Relaxed),
        scores,
        "{route}: warm rescoring"
    );
    println!(
        "cpu_source census/{ORDERED}/{N}/{route}:64 unique calls alloc={} realloc={} free={} bytes={} rescore=0",
        total.allocations, total.reallocations, total.frees, total.requested_bytes
    );
}

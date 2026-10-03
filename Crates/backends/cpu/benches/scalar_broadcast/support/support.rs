//! Matched raw transport, caller census and indexed output ownership.
use std::hint::black_box;
use criterion::{Criterion, Throughput};
use super::sample::{Sample, same};
#[allow(clippy::too_many_lines)] // One four-route fixture owns its cold validation and separate warm census.
pub fn compare<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    profile: &str,
    mut prepared: impl FnMut(&T, &mut [T]),
    mut ordinary: impl FnMut(&T, &mut [T]),
    mut graph: impl FnMut(&T, &mut [T]),
) {
    let sentinel = T::pattern(17);
    let mut output = vec![sentinel; N + 3];
    let mut invoke = |route, input: &T, output: &mut [T]| match route {
        0 => prepared(input, output),
        1 => ordinary(input, output),
        2 => graph(input, output),
        _ => output[..N].fill(*input),
    };
    for seed in [0, 1, 73, 91] {
        let input = T::pattern(seed);
        for route in 0..4 {
            invoke(route, &input, &mut output);
            same(&output[..N], &[input; N]);
            same(&output[N..], &[sentinel; 3]);
        }
    }
    let scores = super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed);
    for route in 0..4 {
        let counts = super::ffi::count_heap(|| {
            for seed in 0..64 {
                let input = T::pattern(1026 + seed);
                invoke(route, black_box(&input), black_box(&mut output));
                black_box(&output);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(
            super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed),
            scores
        );
        same(&output[..N], &[T::pattern(1089); N]);
        same(&output[N..], &[sentinel; 3]);
        println!(
            "census {:?} {profile} N{N} route{route} calls64 allocations0 reallocations0 frees0",
            T::TYPE
        );
    }
    {
        let mut group =
            criterion.benchmark_group(format!("cpu_scalar_broadcast/{:?}/{profile}/N{N}", T::TYPE));
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        let mut seed = 1026;
        for (route, label) in [
            "prepared_annotated_source",
            "ordinary_annotated_source",
            "explicit_graph_diagnostic",
            "native_slice_fill",
        ]
        .into_iter()
        .enumerate()
        {
            group.bench_function(label, |bench| {
                bench.iter(|| {
                    seed += 1;
                    let input = T::pattern(seed);
                    invoke(route, black_box(&input), black_box(&mut output));
                    black_box(&output);
                });
            });
        }
        group.finish();
    }
    assert_eq!(
        super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed),
        scores
    );
}

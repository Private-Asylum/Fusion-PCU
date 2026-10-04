//! Comparable warm H2D/kernel/completion/D2H calls, with changing inputs verified untimed.
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::global::{
    self,
    PcuBackendChoice,
    PcuExecutionPolicy,
    PcuInvocationCandidate,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    HipRuntime,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
};
#[rustfmt::skip]
use std::{
    error::Error,
    sync::atomic::{
        AtomicUsize,
        Ordering,
    },
    time::{
        Duration,
        Instant,
    },
};
#[rustfmt::skip]
use super::{
    native::Native,
    oracle,
    source,
    support,
};

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    super::activity::guard();
    let discovery = RocmDiscovery::new();
    let candidates = support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect();
    let (backend, selected) = support::selection::open_ranked(&discovery, candidates, 256)?;
    let runtime = discovery.open_device(selected.device)?;
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Rocm,
        device: Some(selected.device.id),
        score_invocation: Some(score),
        ..PcuExecutionPolicy::default()
    })?;
    println!("Pageable byte roundtrip device: {}", selected.name);
    println!(
        "All routes include H2D, byte identity, completion, D2H and host publication. Native uses handwritten HIP plus public runtime wrappers with retained private pageable staging. Explicit prepared IR is generated from the same source."
    );
    case::<4096>(criterion, &backend, &runtime)?;
    case::<4_194_304>(criterion, &backend, &runtime)
}

#[allow(clippy::significant_drop_tightening)] // Criterion's group owns all three registrations until finish.
fn case<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
) -> Result<(), Box<dyn Error>> {
    super::activity::guard();
    let mut native = support::cold_once("host-transfer/native prepare", || {
        Native::prepare(runtime, N)
    })?;
    let mut prepared = support::cold_once("host-transfer/source prepare", || {
        source::copy_prepare::<N, _>(backend)
    })?;
    let mut input = vec![0; N];
    let mut output = vec![oracle::SENTINEL; N + oracle::TAIL];
    // Alternating fresh contents and oversized host output catch stale warm-cache bindings
    // and publication past the declared invocation extent before measuring the exact shape.
    for sequence in [0x0102_0304_0506_0708, 0xf1e2_d3c4_b5a6_9788] {
        oracle::refresh(&mut input, sequence);
        output.fill(oracle::SENTINEL);
        source::copy::<N>(&input, &mut output)?;
        oracle::verify(&input, &output);
        output.fill(oracle::SENTINEL);
        prepared(&input, &mut output)?;
        oracle::verify(&input, &output);
        output.fill(oracle::SENTINEL);
        native.execute(&input, &mut output)?;
        oracle::verify(&input, &output);
    }
    output.truncate(N);
    // Warm the exact measured shapes and private native transfer RAM before sampling.
    source::copy::<N>(&input, &mut output)?;
    prepared(&input, &mut output)?;
    native.execute(&input, &mut output)?;
    oracle::verify(&input, &output);
    let scores = SCORES.load(Ordering::Relaxed);
    let mut sequence = 0_u64;
    let mut group = criterion.benchmark_group("rocm_byte_identity_pageable_roundtrip");
    group.throughput(Throughput::Bytes(u64::try_from(N * 2)?));
    group.bench_function(format!("borrowed_source/{N}"), |bench| {
        bench.iter_custom(|iterations| {
            measure(
                iterations,
                &mut sequence,
                &mut input,
                &mut output,
                |input, output| {
                    source::copy::<N>(input, output).expect("source identity failed");
                },
            )
        });
    });
    group.bench_function(format!("explicit_source_prepared_ir/{N}"), |bench| {
        bench.iter_custom(|iterations| {
            measure(
                iterations,
                &mut sequence,
                &mut input,
                &mut output,
                |input, output| {
                    prepared(input, output).expect("prepared identity failed");
                },
            )
        });
    });
    group.bench_function(format!("native_handwritten_hip/{N}"), |bench| {
        bench.iter_custom(|iterations| {
            measure(
                iterations,
                &mut sequence,
                &mut input,
                &mut output,
                |input, output| {
                    native
                        .execute(input, output)
                        .expect("native identity failed");
                },
            )
        });
    });
    group.finish();
    assert_eq!(
        SCORES.load(Ordering::Relaxed),
        scores,
        "warm calls must reuse admission"
    );
    Ok(())
}

fn measure(
    iterations: u64,
    sequence: &mut u64,
    input: &mut [u8],
    output: &mut [u8],
    mut execute: impl FnMut(&[u8], &mut [u8]),
) -> Duration {
    let mut elapsed = Duration::ZERO;
    for _ in 0..iterations {
        *sequence = sequence
            .checked_add(1)
            .expect("unique benchmark call sequence");
        oracle::refresh(input, *sequence);
        output.fill(oracle::SENTINEL);
        let started = Instant::now();
        execute(
            std::hint::black_box(&*input),
            std::hint::black_box(&mut *output),
        );
        elapsed += started.elapsed();
        oracle::verify(input, output);
    }
    elapsed
}

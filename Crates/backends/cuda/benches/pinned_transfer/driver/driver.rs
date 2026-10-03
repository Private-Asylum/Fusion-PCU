//! Complete identity calls pair authoring layers; copy-only diagnostics stay explicitly separate.
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuOwnedDispatchBackend,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaOwnedDispatchBackend,
    CudaRuntime,
};
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
use super::{native::Native, source};

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

pub fn configure(backend: &CudaOwnedDispatchBackend) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(backend.device_identity().device_id()),
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
}

#[allow(clippy::too_many_lines)] // Freeze all six physical staging routes before comparing warm work.
pub fn case<const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    runtime: &CudaRuntime,
) {
    let banks = [vec![17_u8; N], vec![239; N]];
    let pinned = banks.each_ref().map(|input| {
        let mut buffer = runtime.allocate_pinned(N).unwrap();
        buffer.as_bytes_mut().copy_from_slice(input);
        buffer
    });
    let mut output = vec![0; N];
    let mut native = Native::new::<N>(backend, runtime);
    let mut prepared = source::copy_prepare::<N, _>(backend).unwrap();
    for bank in 0..2 {
        source::copy::<N>(&banks[bank], &mut output).unwrap();
        assert_eq!(output, banks[bank]);
        source::copy::<N>(pinned[bank].as_bytes(), &mut output).unwrap();
        assert_eq!(output, banks[bank]);
        prepared(&banks[bank], &mut output).unwrap();
        assert_eq!(output, banks[bank]);
        native.pageable(&banks[bank], &mut output);
        assert_eq!(output, banks[bank]);
        native.pageable(pinned[bank].as_bytes(), &mut output);
        assert_eq!(output, banks[bank]);
        native.pinned(bank);
        assert_eq!(native.pinned_result(), banks[bank]);
    }
    #[cfg(feature = "allocation-census")]
    {
        super::allocations::census(&format!("identity_roundtrip/source/{N}"), || {
            source::copy::<N>(&banks[0], &mut output).unwrap();
        });
        for route in 0..6 {
            let before = SCORES.load(Ordering::Relaxed);
            fusion_pcu_cuda::reset_cuda_api_census();
            let ((), rust) = super::allocations::measure(|| {
                for iteration in 0..64 {
                    let bank = iteration % 2;
                    match route {
                        0 => source::copy::<N>(&banks[bank], &mut output).unwrap(),
                        1 => source::copy::<N>(pinned[bank].as_bytes(), &mut output).unwrap(),
                        2 => prepared(&banks[bank], &mut output).unwrap(),
                        3 => native.pageable(&banks[bank], &mut output),
                        4 => native.pageable(pinned[bank].as_bytes(), &mut output),
                        _ => native.pinned(bank),
                    }
                }
            });
            let api = fusion_pcu_cuda::cuda_api_census();
            assert_eq!(api.kernel_launches, 64);
            assert_eq!(api.symbol_resolutions, 0);
            assert_eq!(api.module_loads, 0);
            assert_eq!(SCORES.load(Ordering::Relaxed), before);
            eprintln!(
                "census/identity_roundtrip/{N}/{route}/64-changing-calls: Rust alloc={} realloc={} frees={} bytes={}; API={api:?}; scores={before}",
                rust.alloc_calls, rust.realloc_calls, rust.dealloc_calls, rust.requested_bytes
            );
            if route == 5 {
                assert_eq!(native.pinned_result(), banks[1]);
            } else {
                assert_eq!(output, banks[1]);
            }
        }
    }
    let before = SCORES.load(Ordering::Relaxed);
    let mut phase = 0;
    let mut group = criterion.benchmark_group("cuda_source_identity_host_device_roundtrip");
    group.throughput(Throughput::Bytes(u64::try_from(N * 2).unwrap()));
    group.bench_function(format!("ordinary_source_pageable/{N}"), |bench| {
        bench.iter(|| {
            phase ^= 1;
            source::copy::<N>(&banks[phase], &mut output).unwrap();
        });
    });
    group.bench_function(format!("ordinary_source_pinned_borrow/{N}"), |bench| {
        bench.iter(|| {
            phase ^= 1;
            source::copy::<N>(pinned[phase].as_bytes(), &mut output).unwrap();
        });
    });
    group.bench_function(format!("explicit_prepared_pageable/{N}"), |bench| {
        bench.iter(|| {
            phase ^= 1;
            prepared(&banks[phase], &mut output).unwrap();
        });
    });
    group.bench_function(format!("native_pageable_same_kernel/{N}"), |bench| {
        bench.iter(|| {
            phase ^= 1;
            native.pageable(&banks[phase], &mut output);
        });
    });
    group.bench_function(
        format!("native_pinned_input_borrow_same_kernel/{N}"),
        |bench| {
            bench.iter(|| {
                phase ^= 1;
                native.pageable(pinned[phase].as_bytes(), &mut output);
            });
        },
    );
    group.bench_function(format!("native_pinned_events_same_kernel/{N}"), |bench| {
        bench.iter(|| {
            phase ^= 1;
            native.pinned(phase);
        });
    });
    assert_eq!(SCORES.load(Ordering::Relaxed), before);
    group.finish();
}

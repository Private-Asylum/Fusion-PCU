//! Retained source, explicit dispatch and native controls share the actual work boundary.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCompletionOutcome,
    PcuOwnedDispatchBackend,
    PcuDispatchIntegerBinaryOp,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaOwnedDispatchBackend,
};
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
use super::{bindings, support};
use super::source;

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

fn bytes(value: u32, count: usize) -> Vec<u8> {
    std::iter::repeat_n(value, count)
        .flat_map(u32::to_ne_bytes)
        .collect()
}

#[allow(clippy::too_many_lines)] // One physical fixture retains all three routes and their storage.
pub fn case<const N: usize>(criterion: &mut Criterion, backend: &CudaOwnedDispatchBackend) {
    let extent = u32::try_from(N).unwrap();
    let prepared = support::prepare(backend, PcuDispatchIntegerBinaryOp::Add, extent, false);
    let hosts = [vec![3_u32; N], vec![29_u32; N]];
    let residents = hosts.each_ref().map(|input| source::retain(input).unwrap());
    let seed = source::retain(&[7]).unwrap();
    let mut destination = source::retain(&vec![0; N]).unwrap();
    let banks = [3, 29].map(|value| {
        let mut buffer = backend.allocate(N * size_of::<u32>()).unwrap();
        buffer.copy_from(&bytes(value, N)).unwrap();
        buffer
    });
    let mut right = backend.allocate(size_of::<u32>()).unwrap();
    right.copy_from(&7_u32.to_ne_bytes()).unwrap();
    let output = backend.allocate(N * size_of::<u32>()).unwrap();
    let mut status = backend.allocate(size_of::<u64>()).unwrap();
    status.copy_from(&u64::MAX.to_le_bytes()).unwrap();
    let declarations = banks
        .each_ref()
        .map(|left| bindings(backend, left, &right, &output));
    let mut sequential = prepared.sequential_checked().unwrap();
    let kernel = prepared.cuda_kernel();
    let stream = prepared.stream_handle();
    let (grid, block) = prepared.launch_geometry();
    let mut word = [0; 8];
    let mut actual = vec![0_u32; N];
    let mut raw = vec![0; N * size_of::<u32>()];
    let mut native = |bank: usize| {
        ffi::launch(
            &kernel,
            &stream,
            grid,
            block,
            [&banks[bank], &right, &output, &status],
        )
        .unwrap()
        .wait()
        .unwrap();
        status.copy_to(&mut word).unwrap();
        assert_eq!(u64::from_le_bytes(word), u64::MAX);
    };
    for bank in 0..2 {
        source::add::<N>(&residents[bank], &seed, &mut destination).unwrap();
        destination.read_into(&mut actual).unwrap();
        assert_eq!(actual, vec![hosts[bank][0] + 7; N]);
        assert_eq!(
            sequential.submit_and_wait(&declarations[bank]).unwrap(),
            PcuCompletionOutcome::Succeeded
        );
        output.copy_to(&mut raw).unwrap();
        assert_eq!(raw, bytes(hosts[bank][0] + 7, N));
        native(bank);
        output.copy_to(&mut raw).unwrap();
        assert_eq!(raw, bytes(hosts[bank][0] + 7, N));
    }
    #[cfg(feature = "allocation-census")]
    super::allocations::census(&format!("checked_dispatch/source/{N}"), || {
        source::add::<N>(&residents[0], &seed, &mut destination).unwrap();
    });
    #[cfg(feature = "allocation-census")]
    for route in 0..3 {
        let before = SCORES.load(Ordering::Relaxed);
        fusion_pcu_cuda::reset_cuda_api_census();
        let ((), rust) = super::allocations::measure(|| {
            for iteration in 0..64 {
                let bank = iteration % 2;
                match route {
                    0 => source::add::<N>(&residents[bank], &seed, &mut destination).unwrap(),
                    1 => assert_eq!(
                        sequential.submit_and_wait(&declarations[bank]).unwrap(),
                        PcuCompletionOutcome::Succeeded
                    ),
                    _ => native(bank),
                }
            }
        });
        let api = fusion_pcu_cuda::cuda_api_census();
        assert_eq!(api.kernel_launches, 64);
        assert_eq!(api.symbol_resolutions, 0);
        assert_eq!(api.module_loads, 0);
        assert_eq!(SCORES.load(Ordering::Relaxed), before);
        eprintln!(
            "census/checked_dispatch/{N}/{route}/64-changing-calls: Rust alloc={} realloc={} frees={} bytes={}; API={api:?}; scores={before}",
            rust.alloc_calls, rust.realloc_calls, rust.dealloc_calls, rust.requested_bytes
        );
        if route == 0 {
            destination.read_into(&mut actual).unwrap();
            assert_eq!(actual, vec![36; N]);
        } else {
            output.copy_to(&mut raw).unwrap();
            assert_eq!(raw, bytes(36, N));
        }
    }
    let before = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group("cuda_checked_u32_add_source_retained");
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    let mut phase = 0;
    group.bench_function(format!("ordinary_source/{N}"), |bench| {
        bench.iter(|| {
            phase ^= 1;
            source::add::<N>(&residents[phase], &seed, &mut destination).unwrap();
        });
    });
    group.bench_function(format!("explicit_sequential/{N}"), |bench| {
        bench.iter(|| {
            phase ^= 1;
            assert_eq!(
                sequential.submit_and_wait(&declarations[phase]).unwrap(),
                PcuCompletionOutcome::Succeeded
            );
        });
    });
    group.bench_function(format!("native_same_lowering/{N}"), |bench| {
        bench.iter(|| {
            phase ^= 1;
            native(phase);
        });
    });
    assert_eq!(SCORES.load(Ordering::Relaxed), before);
    group.finish();
}

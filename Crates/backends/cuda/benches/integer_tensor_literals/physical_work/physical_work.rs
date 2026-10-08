//! Optional physical-work controls; the preallocated combined native route remains primary.
//! Fresh output and split completion isolate two costs. Native status remains 16 bytes and
//! a memset reset, versus the source's 32-byte guarded slab and H2D reset: this is approximate.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuNumericalMode,
};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use std::time::Duration;
use std::time::Instant;
use super::native::Native;

pub fn run(c: &mut Criterion, device: u32, semantics: bool) {
    case::<65>(c, device, semantics);
    case::<4096>(c, device, semantics);
}

#[allow(clippy::too_many_lines)] // One witness keeps source/native semantic and work checks together.
fn case<const N: usize>(c: &mut Criterion, device: u32, semantics: bool) {
    #[cfg(feature = "allocation-census")]
    let _ = (&mut *c, semantics);
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(device),
        numerical_mode: PcuNumericalMode::Strict,
        ..Default::default()
    })
    .unwrap();
    let constant = [1_u32; N];
    let uniform = [2_u32; N];
    let mut input = [1_u32; N];
    let mut observed = vec![77_u32; N + 2];
    let mut native = Native::new::<u32, N>(device);
    #[cfg(feature = "allocation-census")]
    let counter = native.counter();
    let mut routes = [
        "source",
        "native",
        "native_fresh_output",
        "native_split_completion",
        "native_fresh_split_completion",
    ];
    if std::env::var_os("PCU_PHYSICAL_WORK_REVERSE").is_some() {
        routes.reverse();
    }
    for route in routes {
        let mut call = |input: &[u32], out: &mut [u32]| -> Result<Option<(u32, u32)>, ()> {
            if input.len() != N || out.len() < N {
                return Err(());
            }
            match route {
                "source" => match super::source::pipeline(input, &constant, &uniform) {
                    Ok(owner) => {
                        owner.read_into(out).map_err(|_| ())?;
                        Ok(None)
                    }
                    Err(error) => error
                        .arithmetic_fault()
                        .map(|f| {
                            Some((
                                u32::try_from(f.invocation_id).unwrap(),
                                super::oracle::code(f.kind),
                            ))
                        })
                        .ok_or(()),
                },
                "native_fresh_output" => native.call_fresh_output(input, &constant, &uniform, out),
                "native_split_completion" => {
                    native.call_split_completion(input, &constant, &uniform, out)
                }
                "native_fresh_split_completion" => {
                    native.call_fresh_split_completion(input, &constant, &uniform, out)
                }
                _ => native.call(input, &constant, &uniform, out),
            }
        };
        assert_eq!(call(&input, &mut observed), Ok(None));
        verify(&input, &observed);
        for bad in [u32::MAX, u32::MAX / 2] {
            input[N - 1] = bad;
            let previous = observed.clone();
            assert_eq!(
                call(&input, &mut observed),
                Ok(Some((u32::try_from(N - 1).unwrap(), 3)))
            );
            assert_eq!(observed, previous);
            input[N - 1] = 1;
            assert_eq!(call(&input, &mut observed), Ok(None));
            verify(&input, &observed);
        }
        let previous = observed.clone();
        assert_eq!(call(&input[..N - 1], &mut observed), Err(()));
        assert_eq!(call(&input, &mut observed[..N - 1]), Err(()));
        assert_eq!(observed, previous);
        let mut generation = 0_u32;
        let mut warm = || {
            generation = generation.wrapping_add(1);
            input[0] = generation & 0x00ff_ffff;
            let start = Instant::now();
            let result = call(&input, &mut observed);
            let elapsed = start.elapsed();
            assert_eq!(result, Ok(None));
            verify(&input, &observed);
            elapsed
        };
        let _ = warm();
        #[cfg(feature = "allocation-census")]
        {
            let before = fusion_pcu_cuda::cuda_api_census();
            let sdk_before = counter.get();
            let ((), heap) = super::allocations::measure(|| {
                for _ in 0..64 {
                    std::hint::black_box(warm());
                }
            });
            let after = fusion_pcu_cuda::cuda_api_census();
            let sdk = counter.get().delta(sdk_before);
            if route == "source" {
                assert_eq!(
                    after.guarded_chain_submissions - before.guarded_chain_submissions,
                    64
                );
                assert_eq!(after.allocations - before.allocations, 64);
                assert_eq!(after.frees - before.frees, 64);
                assert_eq!(
                    after.host_to_device_copies - before.host_to_device_copies,
                    256
                );
                assert_eq!(
                    after.device_to_host_copies - before.device_to_host_copies,
                    128
                );
                assert_eq!(after.kernel_launches - before.kernel_launches, 128);
                assert_eq!(after.event_waits - before.event_waits, 64);
                assert_eq!(heap.alloc_calls, 64);
            } else {
                let fresh = route.contains("fresh");
                let split = route.contains("split");
                assert_eq!(
                    (sdk.allocations, sdk.frees),
                    if fresh { (64, 64) } else { (0, 0) }
                );
                assert_eq!(
                    (sdk.uploads, sdk.downloads, sdk.resets, sdk.launches),
                    (192, 128, 64, 128)
                );
                assert_eq!(
                    (sdk.event_records, sdk.event_waits),
                    if split { (128, 128) } else { (64, 64) }
                );
                assert_eq!(sdk.uploaded_bytes, u64::try_from(192 * N * 4).unwrap());
                assert_eq!(
                    sdk.downloaded_bytes,
                    u64::try_from(64 * (N * 4 + 16)).unwrap()
                );
                assert_eq!(
                    (heap.alloc_calls, heap.realloc_calls, heap.dealloc_calls),
                    (0, 0, 0)
                );
            }
            println!(
                "cuda-physical-work/{N}/{route}/64-changing-calls: heap={}/{}/{}/{}; before={before:?}; after={after:?}; sdk={sdk:?}",
                heap.alloc_calls, heap.realloc_calls, heap.dealloc_calls, heap.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        if !semantics {
            std::thread::sleep(Duration::from_secs(5));
            super::activity::activity_guard();
            let mut group = c.benchmark_group(format!("cuda_physical_work/{N}/{route}"));
            group.sample_size(20);
            group.warm_up_time(Duration::from_millis(500));
            group.measurement_time(Duration::from_secs(2));
            group.bench_function("full_host_boundary", |b| {
                b.iter_custom(|n| (0..n).map(|_| warm()).sum::<Duration>());
            });
            group.finish();
        }
        println!(
            "cuda-physical-work/{N}/{route}: exact results, tails, both fault stages, retry and short inputs PASS"
        );
    }
    global::clear_thread_cache().unwrap();
}

fn verify(input: &[u32], observed: &[u32]) {
    for (&x, &actual) in input.iter().zip(observed) {
        assert_eq!(actual, (x + 1) * 2);
    }
    assert_eq!(&observed[input.len()..], &[77, 77]);
}

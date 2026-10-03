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
    PcuBindingRef,
    PcuBindingAccess,
    PcuBindingType,
    PcuOwnedBinding,
    PcuValueType,
    PcuDispatchSubmission,
    PcuInvocationShape,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmOwnedDispatchBackend,
};
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
use std::num::NonZeroU32;
use super::source;

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

pub fn configure(backend: &RocmOwnedDispatchBackend) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(backend.device_identity().device_id()),
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
}

fn bytes(value: f32, count: usize) -> Vec<u8> {
    std::iter::repeat_n(value, count)
        .flat_map(f32::to_ne_bytes)
        .collect()
}

#[allow(clippy::too_many_lines)] // One physical fixture retains all three routes and their storage.
pub fn case<const N: usize>(criterion: &mut Criterion, backend: &RocmOwnedDispatchBackend) {
    let extent = u32::try_from(N).unwrap();
    let declaration = source::add_bindings();
    let builder = source::add_ir::<N>(&declaration).unwrap();
    let prepared = builder
        .with_ir(|kernel| {
            backend.prepare_dispatch(PcuDispatchSubmission {
                kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(extent).unwrap()),
            })
        })
        .unwrap();
    let hosts = [vec![1.0_f32; N], vec![2.0; N]];
    let residents = hosts.each_ref().map(|input| source::retain(input).unwrap());
    let seed = source::retain(&[0.25]).unwrap();
    let mut destination = source::retain(&vec![0.0; N]).unwrap();
    let banks = [1.0, 2.0].map(|value| {
        let mut buffer = backend.allocate(N * size_of::<f32>()).unwrap();
        buffer.copy_from(&bytes(value, N)).unwrap();
        buffer
    });
    let mut right = backend.allocate(size_of::<f32>()).unwrap();
    right.copy_from(&0.25_f32.to_ne_bytes()).unwrap();
    let output = backend.allocate(N * size_of::<f32>()).unwrap();
    let mut status = backend.allocate(size_of::<u64>()).unwrap();
    status.copy_from(&u64::MAX.to_le_bytes()).unwrap();
    let declarations = banks
        .each_ref()
        .map(|left| bindings(backend, left, &right, &output));
    let mut sequential = prepared.sequential_checked().unwrap();
    let kernel = prepared.hip_kernel();
    let stream = prepared.stream_handle();
    let (grid, block) = prepared.launch_geometry();
    let mut word = [0; 8];
    let mut actual = vec![0.0_f32; N];
    let mut raw = vec![0; N * size_of::<f32>()];
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
        assert_eq!(actual, vec![hosts[bank][0] + 0.25; N]);
        assert_eq!(
            sequential.submit_and_wait(&declarations[bank]).unwrap(),
            PcuCompletionOutcome::Succeeded
        );
        output.copy_to(&mut raw).unwrap();
        assert_eq!(raw, bytes(hosts[bank][0] + 0.25, N));
        native(bank);
        output.copy_to(&mut raw).unwrap();
        assert_eq!(raw, bytes(hosts[bank][0] + 0.25, N));
    }
    #[cfg(feature = "allocation-census")]
    for route in 0..3 {
        let before = SCORES.load(Ordering::Relaxed);
        fusion_pcu_rocm::reset_rocm_api_census();
        let capture = super::alloc::AllocationCapture::start();
        let initial = super::alloc::AllocationCapture::snapshot();
        {
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
        }
        let rust = super::alloc::AllocationCapture::finish().since(initial);
        drop(capture);
        let api = fusion_pcu_rocm::rocm_api_census();
        assert_eq!(api.kernel_launches, 64);
        assert_eq!(api.symbol_resolutions, 0);
        assert_eq!(api.module_loads, 0);
        assert_eq!(SCORES.load(Ordering::Relaxed), before);
        eprintln!(
            "census/tensor_uniform/{N}/{route}/64-changing-calls: Rust alloc={} realloc={} frees={} bytes={}; API={api:?}; scores={before}",
            rust.alloc_calls, rust.realloc_calls, rust.dealloc_calls, rust.requested_bytes
        );
        if route == 0 {
            destination.read_into(&mut actual).unwrap();
            assert_eq!(actual, vec![2.25; N]);
        } else {
            output.copy_to(&mut raw).unwrap();
            assert_eq!(raw, bytes(2.25, N));
        }
    }
    let before = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group("rocm_uniform_add_source_retained");
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

fn bindings(
    backend: &RocmOwnedDispatchBackend,
    left: &fusion_pcu_rocm::DeviceBuffer,
    right: &fusion_pcu_rocm::DeviceBuffer,
    output: &fusion_pcu_rocm::DeviceBuffer,
) -> [PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>; 3] {
    [
        (0, left, PcuBindingAccess::ReadOnly),
        (1, right, PcuBindingAccess::ReadOnly),
        (2, output, PcuBindingAccess::ReadWrite),
    ]
    .map(|(index, resource, access)| {
        backend
            .binding(
                PcuBindingRef::new(0, index),
                access,
                PcuBindingType::Value(PcuValueType::f32()),
                resource.clone(),
            )
            .unwrap()
    })
}

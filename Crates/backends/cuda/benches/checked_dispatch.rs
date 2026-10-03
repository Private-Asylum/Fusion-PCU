//! Matched native-driver and checked PCU retained dispatch benchmark.
extern crate pcu_facade as fusion_pcu;

#[cfg(feature = "allocation-census")]
#[path = "support/allocations/allocations.rs"]
mod allocations;
#[path = "checked_dispatch/source/source.rs"]
mod source;
#[path = "checked_dispatch/fixture/fixture.rs"]
mod source_fixture;

#[path = "support/checked_dispatch.rs"]
mod support;

#[rustfmt::skip]
use criterion::{
    BenchmarkGroup,
    Criterion,
    measurement::WallTime,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuOwnedBinding,
    PcuCompletionOutcome,
    PcuDispatchIntegerBinaryOp,
    PcuOwnedCompletion,
    PcuValueType,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaGraphDispatch,
    CudaKernelArgument,
    CudaNativeGraph,
    CudaPreparedDispatch,
    CudaOwnedDispatchBackend,
    DeviceBuffer,
};
#[path = "support/activity.rs"]
mod activity;
use activity::activity_guard;

fn bindings(
    backend: &CudaOwnedDispatchBackend,
    left: &DeviceBuffer,
    right: &DeviceBuffer,
    output: &DeviceBuffer,
) -> [PcuOwnedBinding<DeviceBuffer>; 3] {
    [
        (0, left, PcuBindingAccess::ReadOnly),
        (1, right, PcuBindingAccess::ReadOnly),
        (2, output, PcuBindingAccess::WriteOnly),
    ]
    .map(|(index, buffer, access)| {
        backend
            .binding(
                PcuBindingRef::new(0, index),
                access,
                PcuBindingType::Value(PcuValueType::u32()),
                buffer.clone(),
            )
            .unwrap()
    })
}

fn bench_graph(
    group: &mut BenchmarkGroup<'_, WallTime>,
    prepared: &CudaPreparedDispatch,
    bindings: &[PcuOwnedBinding<DeviceBuffer>; 3],
    extent: u32,
) {
    let mut graph = CudaNativeGraph::capture(&[CudaGraphDispatch {
        dispatch: prepared,
        bindings,
    }])
    .unwrap();
    let mut actual = vec![0; usize::try_from(extent).unwrap() * size_of::<u32>()];
    for value in [13_u32, 47] {
        let source: Vec<u8> = std::iter::repeat_n(value, usize::try_from(extent).unwrap())
            .flat_map(u32::to_ne_bytes)
            .collect();
        graph
            .refresh_input(&bindings[0].resource, 0, &source)
            .unwrap();
        graph
            .refresh_input(&bindings[1].resource, 0, &7_u32.to_ne_bytes())
            .unwrap();
        assert_eq!(
            graph.replay_checked_and_wait().unwrap(),
            PcuCompletionOutcome::Succeeded
        );
        graph
            .readback(&bindings[2].resource, 0, &mut actual)
            .unwrap();
        let expected: Vec<u8> = std::iter::repeat_n(value + 7, usize::try_from(extent).unwrap())
            .flat_map(u32::to_ne_bytes)
            .collect();
        assert_eq!(actual, expected);
    }
    // This route includes captured sentinel reset, graph launch, stream wait and status readback.
    // Its reset scheduling differs from the retained-success-sentinel direct routes above.
    group.bench_function(format!("pcu_native_graph_reset/{extent}"), |bench| {
        bench.iter(|| {
            assert_eq!(
                graph.replay_checked_and_wait().unwrap(),
                PcuCompletionOutcome::Succeeded
            );
        });
    });
}

fn checked_dispatch(criterion: &mut Criterion) {
    activity_guard();
    let (_discovery, backend) = support::selected_device();
    source_fixture::configure(&backend);
    source_fixture::case::<256>(criterion, &backend);
    source_fixture::case::<65536>(criterion, &backend);
    let mut group = criterion.benchmark_group("cuda_checked_u32_add_retained");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(3));
    for extent in [256_u32, 65536] {
        activity::compute_owner_guard();
        let prepared = support::prepare(&backend, PcuDispatchIntegerBinaryOp::Add, extent, false);
        let byte_len = usize::try_from(extent).unwrap() * size_of::<u32>();
        let mut left = backend.allocate(byte_len).unwrap();
        let mut right = backend.allocate(size_of::<u32>()).unwrap();
        let output = backend.allocate(byte_len).unwrap();
        let mut status = backend.allocate(size_of::<u64>()).unwrap();
        let bindings = bindings(&backend, &left, &right, &output);
        let kernel = prepared.cuda_kernel();
        let stream = prepared.stream_handle();
        let (grid, block) = prepared.launch_geometry();
        let mut sequential = prepared.sequential_checked().unwrap();
        let mut actual = vec![0; byte_len];
        let mut word = [0; 8];
        // Change input and prove both routes outside measurement, using the same generated kernel.
        for value in [3_u32, 29] {
            let source: Vec<u8> = std::iter::repeat_n(value, usize::try_from(extent).unwrap())
                .flat_map(u32::to_ne_bytes)
                .collect();
            left.copy_from(&source).unwrap();
            right.copy_from(&7_u32.to_ne_bytes()).unwrap();
            status.copy_from(&u64::MAX.to_le_bytes()).unwrap();
            let arguments = [
                CudaKernelArgument::Buffer(&left),
                CudaKernelArgument::Buffer(&right),
                CudaKernelArgument::Buffer(&output),
                CudaKernelArgument::Buffer(&status),
            ];
            // SAFETY: this is the exact generated three-u32-buffer + u64-status ABI, with
            // identical retained allocations and launch geometry to the checked PCU route.
            unsafe { kernel.launch(&stream, grid, block, 0, &arguments) }
                .unwrap()
                .wait()
                .unwrap();
            status.copy_to(&mut word).unwrap();
            assert_eq!(u64::from_le_bytes(word), u64::MAX);
            output.copy_to(&mut actual).unwrap();
            let expected: Vec<u8> =
                std::iter::repeat_n(value + 7, usize::try_from(extent).unwrap())
                    .flat_map(u32::to_ne_bytes)
                    .collect();
            assert_eq!(actual, expected);
            assert_eq!(
                sequential.submit_and_wait(&bindings).unwrap(),
                PcuCompletionOutcome::Succeeded
            );
            output.copy_to(&mut actual).unwrap();
            assert_eq!(actual, expected);
        }
        let arguments = [
            CudaKernelArgument::Buffer(&left),
            CudaKernelArgument::Buffer(&right),
            CudaKernelArgument::Buffer(&output),
            CudaKernelArgument::Buffer(&status),
        ];
        group.throughput(Throughput::Elements(u64::from(extent)));
        // Both retained routes reuse a terminally observed sentinel. Each iteration includes
        // launch, event completion and status readback; neither includes input/output transfers.
        group.bench_function(format!("native_driver/{extent}"), |bench| {
            bench.iter(|| {
                // SAFETY: the validated fixture above establishes ABI, allocation sizes and geometry.
                unsafe { kernel.launch(&stream, grid, block, 0, &arguments) }
                    .unwrap()
                    .wait()
                    .unwrap();
                status.copy_to(&mut word).unwrap();
                assert_eq!(u64::from_le_bytes(word), u64::MAX);
            });
        });
        group.bench_function(format!("pcu_sequential/{extent}"), |bench| {
            bench.iter(|| {
                assert_eq!(
                    sequential.submit_and_wait(&bindings).unwrap(),
                    PcuCompletionOutcome::Succeeded
                );
            });
        });
        group.bench_function(format!("pcu_fresh_status/{extent}"), |bench| {
            bench.iter(|| {
                let mut completion = prepared.submit(&bindings).unwrap();
                assert_eq!(
                    PcuOwnedCompletion::wait(&mut completion).unwrap(),
                    PcuCompletionOutcome::Succeeded
                );
            });
        });
        bench_graph(&mut group, &prepared, &bindings, extent);
    }
    group.finish();
}

criterion_group!(benches, checked_dispatch);
criterion_main!(benches);

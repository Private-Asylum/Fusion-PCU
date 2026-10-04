//! Explicit source and independent typed IR owners retain their originating SDK ledger.
#[path = "../../prepared/composed/tests/graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use fusion_pcu_vulkan::{with_api_insights_scope,PcuVulkanApiInsights,PcuVulkanApiPoint as Point,PcuVulkanBackend,PcuVulkanCountOnlyClock};
#[rustfmt::skip]
use pcu_facade::{pcu,PcuBindingRef,PcuFloatUnderflowPolicy,PcuHostArgument,PcuPreparedHostKernel,PcuRangePolicy};
use std::rc::Rc;
#[pcu(invocations = 65, crate_path = ::pcu_facade)]
fn expression(output: &mut [f32], input: &[f32]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
fn counts(ledger: &PcuVulkanApiInsights) -> [u64; 96] {
    ledger.records().map(|record| {
        assert_eq!(
            (record.hits, record.inclusive_ticks, record.exclusive_ticks),
            (0, 0, 0)
        );
        record.count
    })
}
#[test]
#[ignore = "requires actual Vulkan compute device"]
#[allow(clippy::too_many_lines)] // One attachment lifecycle keeps every native owner observable through terminal destruction.
fn source_and_hand_ir_warm_calls_keep_original_attachment_and_release_all_owners() {
    let first = Rc::new(PcuVulkanApiInsights::new(PcuVulkanCountOnlyClock));
    let other = Rc::new(PcuVulkanApiInsights::new(PcuVulkanCountOnlyClock));
    let backend = PcuVulkanBackend::with_api_insights(Rc::clone(&first)).unwrap();
    let mut source = expression_prepare(&backend).unwrap();
    let mut graph = graph::prepare::<f32, 65, _>(
        &backend,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    let cold = counts(&first);
    for point in [
        Point::LoaderLoad,
        Point::CreateInstance,
        Point::CreateDevice,
        Point::CreateBuffer,
        Point::AllocateMemory,
        Point::MapMemory,
        Point::CreateShaderModule,
        Point::CreateComputePipelines,
        Point::CreateDescriptorPool,
        Point::AllocateDescriptorSets,
        Point::CreateCommandPool,
        Point::CreateFence,
    ] {
        assert!(cold[point.index()] > 0, "missing cold SDK point {point:?}");
    }
    assert_eq!(cold[Point::CreateShaderModule.index()], 2);
    let other_before = counts(&other);
    with_api_insights_scope(Rc::clone(&other), || {
        for index in 0..64_u32 {
            let value = if index & 1 == 0 { 0.5 } else { 0.25 };
            let input = [value; 65];
            let mut output = [17_f32; 68];
            source(&mut output, &input).unwrap();
            let mut graph_output = [17_f32; 68];
            graph
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), &input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut graph_output),
                ])
                .unwrap();
            assert_eq!(output.map(f32::to_bits), graph_output.map(f32::to_bits));
            assert!(
                output[..65]
                    .iter()
                    .all(|actual| actual.to_bits() == ((value + value) * value).to_bits())
            );
            assert_eq!(output[65..], [17.; 3]);
        }
    });
    assert_eq!(counts(&other), other_before);
    let warm = counts(&first);
    for point in [
        Point::CreateBuffer,
        Point::AllocateMemory,
        Point::FreeMemory,
        Point::MapMemory,
        Point::UnmapMemory,
        Point::CreateShaderModule,
        Point::CreateComputePipelines,
        Point::CreateDescriptorPool,
        Point::AllocateDescriptorSets,
        Point::UpdateDescriptorSets,
        Point::CreateCommandPool,
        Point::AllocateCommandBuffers,
        Point::CreateFence,
    ] {
        assert_eq!(
            warm[point.index()],
            cold[point.index()],
            "warm SDK work at {point:?}"
        );
    }
    assert_eq!(
        warm[Point::QueueSubmit.index()] + warm[Point::QueueSubmit2.index()]
            - cold[Point::QueueSubmit.index()]
            - cold[Point::QueueSubmit2.index()],
        128
    );
    assert_eq!(
        warm[Point::WaitForFences.index()] - cold[Point::WaitForFences.index()],
        128
    );
    assert_eq!(
        warm[Point::ResetFences.index()] - cold[Point::ResetFences.index()],
        128
    );
    drop(source);
    drop(graph);
    drop(backend);
    let terminal = counts(&first);
    for (create, destroy) in [
        (Point::CreateBuffer, Point::DestroyBuffer),
        (Point::AllocateMemory, Point::FreeMemory),
        (Point::MapMemory, Point::UnmapMemory),
        (Point::CreateShaderModule, Point::DestroyShaderModule),
        (Point::CreateComputePipelines, Point::DestroyPipeline),
        (Point::CreateDescriptorPool, Point::DestroyDescriptorPool),
        (Point::CreateCommandPool, Point::DestroyCommandPool),
        (Point::CreateFence, Point::DestroyFence),
        (Point::CreateDevice, Point::DestroyDevice),
        (Point::CreateInstance, Point::DestroyInstance),
    ] {
        assert_eq!(
            terminal[create.index()],
            terminal[destroy.index()],
            "SDK lifecycle {create:?}/{destroy:?}"
        );
    }
    assert_eq!(Rc::strong_count(&first), 1, "temporary guard ledger leaked");
    assert_eq!(Rc::strong_count(&other), 1);
    println!(
        "API source/hand-IR: 128 submissions/waits, zero warm create/allocate/map/update, balanced terminal owners; cold={cold:?} terminal={terminal:?}"
    );
}

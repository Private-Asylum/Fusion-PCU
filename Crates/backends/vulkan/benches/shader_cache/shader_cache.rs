//! Cold artifact policy and retained source/graph/native heap observations, without latency claims.
extern crate pcu_facade as fusion_pcu;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../composed_float_maps/ffi/ffi.rs"]
#[allow(dead_code)] // Retain the separately qualified native controls and stage timers.
mod ffi;
#[path = "../../prepared/composed/tests/graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use pcu_facade::{pcu,PcuBindingRef,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuHostArgument,PcuPreparedHostKernel,PcuScalarType};
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanShaderCachePolicy,PcuVulkanShaderDiskConfig,PcuVulkanShaderSource};
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[pcu(invocations = 65, crate_path = ::pcu_facade)]
fn expression(output: &mut [f32], input: &[f32]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
fn main() {
    let (backend, identity) = device::selected();
    if let Some(directory) = std::env::var_os("PCU_COMPOSED_PACKAGES") {
        backend.configure_shader_source(PcuVulkanShaderSource::ExternalComposed {
            directory: directory.into(),
            retain_in_memory: true,
        });
    }
    let root = std::env::temp_dir().join(format!("pcu-shader-cache-census-{}", std::process::id()));
    assert!(!root.exists());
    let mut native = ffi::NativeComposed::new(identity, 65, 0, 0, PcuScalarType::F32).unwrap();
    native.assert_policy(
        PcuScalarType::F32,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    let input = [0.5_f32; 65];
    for policy in [
        PcuVulkanShaderCachePolicy::Disabled,
        PcuVulkanShaderCachePolicy::MemoryOnly,
        PcuVulkanShaderCachePolicy::Disk(PcuVulkanShaderDiskConfig {
            directory: root.clone(),
            retain_in_memory: false,
            rebuild_invalid: false,
        }),
    ] {
        let label = format!("{policy:?}");
        backend.configure_shader_cache(policy);
        let mut source = expression_prepare(&backend).unwrap();
        let mut graph = graph::prepare::<f32, 65, _>(
            &backend,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Reject,
        );
        // Reprepare the genuine source to exercise a cold artifact hit where enabled.
        let mut repeated = expression_prepare(&backend).unwrap();
        let cold = backend.shader_cache_stats();
        if label.starts_with("MemoryOnly") {
            assert_eq!(cold.memory_hits, 1);
        }
        if label.starts_with("Disk") {
            assert_eq!((cold.disk_hits, cold.disk_misses), (1, 2));
        }
        let mut output = [17_f32; 68];
        let mut graph_output = output;
        let mut native_output = output;
        let source_heap = ffi::count_heap(|| {
            source(&mut output, &input).unwrap();
        });
        let graph_heap = ffi::count_heap(|| {
            graph
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), &input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut graph_output),
                ])
                .unwrap();
        });
        let native_heap = ffi::count_heap(|| {
            assert!(
                native
                    .call(ffi::bytes(&input), ffi::bytes_mut(&mut native_output))
                    .unwrap()
                    .is_none()
            );
        });
        repeated(&mut output, &input).unwrap();
        assert_eq!(output.map(f32::to_bits), graph_output.map(f32::to_bits));
        assert_eq!(output.map(f32::to_bits), native_output.map(f32::to_bits));
        assert!(
            output[..65]
                .iter()
                .all(|value| value.to_bits() == 0.5_f32.to_bits())
        );
        assert!(
            output[65..]
                .iter()
                .all(|value| value.to_bits() == 17_f32.to_bits())
        );
        assert_eq!(backend.shader_cache_stats(), cold);
        for heap in [&source_heap, &graph_heap, &native_heap] {
            assert_eq!(
                (heap.allocations, heap.reallocations, heap.frees),
                (0, 0, 0)
            );
        }
        let counts = |heap: &ffi::HeapCounts| (heap.allocations, heap.reallocations, heap.frees);
        println!(
            "policy={label} cold={cold:?} warm-source={:?} warm-graph={:?} warm-native={:?}",
            counts(&source_heap),
            counts(&graph_heap),
            counts(&native_heap)
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

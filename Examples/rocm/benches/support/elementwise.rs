//! Shared end-to-end Criterion runner for unary and binary f32 tensor maps.

use std::{
    error::Error,
    ffi::CStr,
    hint::black_box,
};

use criterion::{
    Criterion,
    Throughput,
};
use fusion_pcu::PcuOwnedDispatchMemorySession;
use fusion_pcu_rocm::{
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
    compile_hip_source_for_device,
};
use fusion_pcu_tensor::{
    Graph,
    Tensor,
    TensorError,
    ValueId,
};

use crate::support::{
    self,
    selection,
};

/// Workload-specific composition supplied by one small bench target.
pub struct ElementwiseCase {
    pub name: &'static str,
    pub source: &'static str,
    pub kernel_name: &'static CStr,
    pub input_count: usize,
    pub values: fn(usize) -> Vec<Vec<f32>>,
    pub expected: fn(&[Vec<f32>]) -> Vec<f32>,
    pub graph: fn(&mut Graph, &[ValueId]) -> Result<ValueId, TensorError>,
}

/// Measure the same allocation/upload/submission/completion/readback boundary for both routes.
pub fn run(criterion: &mut Criterion, case: &ElementwiseCase) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let mut failures = Vec::new();
    for candidate in support::selected_candidates(&discovery)? {
        match run_on(criterion, case, &discovery, &candidate) {
            Ok(()) => return Ok(()),
            Err(error) => failures.push(format!("{}: {error}", candidate.name)),
        }
    }
    Err(format!(
        "no ROCm device completed {} benchmark: {}",
        case.name,
        failures.join("; ")
    )
    .into())
}

fn run_on(
    criterion: &mut Criterion,
    case: &ElementwiseCase,
    discovery: &RocmDiscovery,
    selected: &selection::Candidate,
) -> Result<(), Box<dyn Error>> {
    let runtime = discovery.open_device(selected.device)?;
    let image = support::cold_once("native HIP compilation", || {
        compile_hip_source_for_device(&runtime, case.source)
    })?;
    let module = runtime.load_module(&image)?;
    let kernel = module.function(case.kernel_name)?;
    let stream = runtime.create_stream()?;
    let session = RocmOwnedDispatchBackend::open(discovery, selected.device, 64)?;
    let assessor = RocmTensorAssessor::new(&session)?;
    println!("Device: {}; workload: {}", selected.name, case.name);
    for size in [65_usize, 1_048_576] {
        let host_values = (case.values)(size);
        if host_values.len() != case.input_count
            || host_values.iter().any(|values| values.len() != size)
        {
            return Err("elementwise case supplied invalid input shape".into());
        }
        let expected = (case.expected)(&host_values);
        if expected.len() != size {
            return Err("elementwise case supplied invalid output shape".into());
        }
        let mut graph = Graph::default();
        let mut inputs = Vec::with_capacity(case.input_count);
        let mut ids = Vec::with_capacity(case.input_count);
        for values in &host_values {
            let id = graph.input([size])?;
            ids.push(id);
            inputs.push((id, Tensor::new([size], values.clone())?));
        }
        let output = (case.graph)(&mut graph, &ids)?;
        let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let cold = support::cold_once(&format!("{} {size} PCU cold", case.name), || {
            assessor.execute_graph(&graph, &inputs, output, selected.pool, &mut memory)
        })?;
        verify(&expected, cold.data())?;
        verify(
            &expected,
            &native_execute(&runtime, &kernel, &stream, &host_values)?,
        )?;

        {
            let mut group = criterion.benchmark_group(format!("tensor_{}/{}", case.name, size));
            group.throughput(Throughput::Elements(u64::try_from(size)?));
            group.bench_function("pcu", |bencher| {
                bencher.iter(|| {
                    black_box(
                        assessor
                            .execute_graph(&graph, &inputs, output, selected.pool, &mut memory)
                            .expect("PCU elementwise execution failed"),
                    )
                });
            });
            group.bench_function("native_hip", |bencher| {
                bencher.iter(|| {
                    black_box(
                        native_execute(&runtime, &kernel, &stream, &host_values)
                            .expect("native HIP elementwise execution failed"),
                    )
                });
            });
            group.finish();
        }

        verify(
            &expected,
            assessor
                .execute_graph(&graph, &inputs, output, selected.pool, &mut memory)?
                .data(),
        )?;
        verify(
            &expected,
            &native_execute(&runtime, &kernel, &stream, &host_values)?,
        )?;
    }
    Ok(())
}

fn verify(expected: &[f32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    if expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| expected.to_bits() == actual.to_bits())
    {
        Ok(())
    } else {
        Err("elementwise PCU/native output mismatch".into())
    }
}

fn native_execute(
    runtime: &HipRuntime,
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    inputs: &[Vec<f32>],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let count = inputs.first().ok_or("native HIP inputs absent")?.len();
    let bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or("byte size overflow")?;
    let mut resources = Vec::with_capacity(inputs.len());
    for values in inputs {
        let mut resource = runtime.allocate(bytes)?;
        resource.copy_from(bytemuck::cast_slice(values))?;
        resources.push(resource);
    }
    let output = runtime.allocate(bytes)?;
    let count_u32 = u32::try_from(count)?;
    let count_bytes = count_u32.to_ne_bytes();
    let mut arguments = resources
        .iter()
        .map(HipKernelArgument::Buffer)
        .collect::<Vec<_>>();
    arguments.push(HipKernelArgument::Buffer(&output));
    arguments.push(HipKernelArgument::Bytes(&count_bytes));
    // SAFETY: Each workload source has exactly input_count f32 pointers followed by an f32
    // output pointer and u32 count. Every buffer spans count elements, padded lanes are guarded,
    // and completion is awaited before resources can be released.
    #[allow(unsafe_code)]
    let mut completion = unsafe {
        kernel.launch(
            stream,
            [count_u32.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &arguments,
        )?
    };
    completion.wait()?;
    let mut result = vec![0.0_f32; count];
    output.copy_to(bytemuck::cast_slice_mut(&mut result))?;
    Ok(result)
}

//! Compare PCU tensor mean-squared error with the equivalent HIP and rocBLAS route.

mod support;

use std::{
    error::Error,
    hint::black_box,
};

use criterion::{
    criterion_group,
    criterion_main,
    Criterion,
    Throughput,
};
use fusion_pcu::PcuOwnedDispatchMemorySession;
use fusion_pcu_rocm::{
    compile_hip_source_for_device,
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    Rocblas,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
};
use fusion_pcu_tensor::{
    Graph,
    Tensor,
};

const SOURCE: &str = r#"
extern "C" __global__ void native_squared_difference(
    const float *prediction, const float *target, float *squared, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float difference = prediction[id] - target[id];
        squared[id] = difference * difference;
    }
}
"#;

struct NativeRoute {
    runtime: HipRuntime,
    kernel: HipKernel,
    stream: HipStreamHandle,
    blas: Rocblas,
}

impl NativeRoute {
    fn execute(&self, prediction: &[f32], target: &[f32]) -> Result<f32, Box<dyn Error>> {
        let count = prediction.len();
        if count == 0 || target.len() != count {
            return Err("MSE inputs must be nonempty and have equal lengths".into());
        }
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or("MSE byte size overflow")?;
        let mut prediction_buffer = self.runtime.allocate(bytes)?;
        prediction_buffer.copy_from(bytemuck::cast_slice(prediction))?;
        let mut target_buffer = self.runtime.allocate(bytes)?;
        target_buffer.copy_from(bytemuck::cast_slice(target))?;
        self.execute_resident(&prediction_buffer, &target_buffer, count)
    }

    fn execute_resident(
        &self,
        prediction: &DeviceBuffer,
        target: &DeviceBuffer,
        count: usize,
    ) -> Result<f32, Box<dyn Error>> {
        if count == 0 {
            return Err("MSE inputs must be nonempty".into());
        }
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or("MSE byte size overflow")?;
        let squared = self.runtime.allocate(bytes)?;
        let output = self.runtime.allocate(size_of::<f32>())?;
        let count_u32 = u32::try_from(count)?;
        let count_bytes = count_u32.to_ne_bytes();
        let arguments = [
            HipKernelArgument::Buffer(prediction),
            HipKernelArgument::Buffer(target),
            HipKernelArgument::Buffer(&squared),
            HipKernelArgument::Bytes(&count_bytes),
        ];
        // SAFETY: The kernel ABI is three f32 device pointers and one u32 count. Each input and
        // output spans count elements, padded lanes are guarded, and completion precedes reuse.
        #[allow(unsafe_code)]
        let mut completion = unsafe {
            self.kernel.launch(
                &self.stream,
                [count_u32.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &arguments,
            )?
        };
        completion.wait()?;
        // Both benchmark sizes are at most 2^20, so their integer divisors are exactly f32.
        #[allow(clippy::cast_precision_loss)]
        let reciprocal = 1.0 / count as f32;
        self.blas
            .sasum_scaled(count, &squared, 1, reciprocal, &output)?;
        let mut result = 0.0_f32;
        output.copy_to(bytemuck::bytes_of_mut(&mut result))?;
        Ok(result)
    }
}

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("paired ROCm MSE benchmark failed");
}

fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let mut failures = Vec::new();
    for candidate in support::selected_candidates(&discovery)? {
        match run_on(criterion, &discovery, &candidate) {
            Ok(()) => return Ok(()),
            Err(error) => failures.push(format!("{}: {error}", candidate.name)),
        }
    }
    Err(format!(
        "no ROCm device completed MSE benchmark: {}",
        failures.join("; ")
    )
    .into())
}

#[allow(clippy::too_many_lines)] // Keeps paired cold, resident, and post-sample checks together.
fn run_on(
    criterion: &mut Criterion,
    discovery: &RocmDiscovery,
    selected: &support::selection::Candidate,
) -> Result<(), Box<dyn Error>> {
    let runtime = discovery.open_device(selected.device)?;
    let image = support::cold_once("native HIP squared-difference compilation", || {
        compile_hip_source_for_device(&runtime, SOURCE)
    })?;
    let module = runtime.load_module(&image)?;
    let native = NativeRoute {
        kernel: module.function(c"native_squared_difference")?,
        stream: runtime.create_stream()?,
        blas: Rocblas::new(&runtime)?,
        runtime,
    };
    verify_reduction_contract(&native.runtime, &native.blas)?;
    // Match the native HIP squared-difference launch geometry.
    let session = RocmOwnedDispatchBackend::open(discovery, selected.device, 256)?;
    let assessor = RocmTensorAssessor::new(&session)?;
    println!("Device: {}; workload: MSE", selected.name);
    for count in [65_usize, 1_048_576] {
        let prediction = (0..count)
            .map(|index| {
                if index.is_multiple_of(2) {
                    1.5_f32
                } else {
                    0.5_f32
                }
            })
            .collect::<Vec<_>>();
        let target = vec![0.5_f32; count];
        let mut graph = Graph::default();
        let prediction_id = graph.input([count])?;
        let target_id = graph.input([count])?;
        let loss = graph.mean_squared_error(prediction_id, target_id)?;
        let inputs = [
            (prediction_id, Tensor::new([count], prediction.clone())?),
            (target_id, Tensor::new([count], target.clone())?),
        ];
        let reference = graph.evaluate(&inputs)?.value(loss)?.clone();
        let mut memory = PcuOwnedDispatchMemorySession::memory_provider(&session, selected.pool);
        let prepared = assessor.prepare_graph(&graph, loss)?;
        let resident_prediction =
            assessor.upload_input(&inputs[0].1, selected.pool, &mut memory)?;
        let resident_target = assessor.upload_input(&inputs[1].1, selected.pool, &mut memory)?;
        let resident_inputs = [
            (prediction_id, &resident_prediction),
            (target_id, &resident_target),
        ];
        let mut native_prediction = native.runtime.allocate(count * size_of::<f32>())?;
        native_prediction.copy_from(bytemuck::cast_slice(&prediction))?;
        let mut native_target = native.runtime.allocate(count * size_of::<f32>())?;
        native_target.copy_from(bytemuck::cast_slice(&target))?;
        let cold = support::cold_once(&format!("MSE {count} PCU cold"), || {
            assessor.execute_graph(&graph, &inputs, loss, selected.pool, &mut memory)
        })?;
        verify(reference.data(), cold.data())?;
        verify(reference.data(), &[native.execute(&prediction, &target)?])?;
        let resident_cold = support::cold_once(&format!("MSE {count} PCU resident cold"), || {
            assessor.execute_prepared_with_resources(
                &prepared,
                &resident_inputs,
                selected.pool,
                &mut memory,
            )
        })?;
        verify(reference.data(), resident_cold.data())?;
        verify(
            reference.data(),
            &[native.execute_resident(&native_prediction, &native_target, count)?],
        )?;
        {
            let mut group = criterion.benchmark_group(format!("tensor_mse/{count}"));
            group.throughput(Throughput::Elements(u64::try_from(count)?));
            group.bench_function("pcu", |bencher| {
                bencher.iter(|| {
                    black_box(
                        assessor
                            .execute_graph(&graph, &inputs, loss, selected.pool, &mut memory)
                            .expect("PCU MSE execution failed"),
                    );
                });
            });
            group.bench_function("pcu_prepared", |bencher| {
                bencher.iter(|| {
                    black_box(
                        assessor
                            .execute_prepared(&prepared, &inputs, selected.pool, &mut memory)
                            .expect("prepared PCU MSE execution failed"),
                    );
                });
            });
            group.bench_function("native_hip_rocblas", |bencher| {
                bencher.iter(|| {
                    black_box(
                        native
                            .execute(&prediction, &target)
                            .expect("native MSE execution failed"),
                    );
                });
            });
            group.finish();
        }
        {
            let mut group = criterion.benchmark_group(format!("tensor_mse_resident/{count}"));
            group.throughput(Throughput::Elements(u64::try_from(count)?));
            group.bench_function("pcu_resident_inputs", |bencher| {
                bencher.iter(|| {
                    black_box(
                        assessor
                            .execute_prepared_with_resources(
                                &prepared,
                                &resident_inputs,
                                selected.pool,
                                &mut memory,
                            )
                            .expect("resident PCU MSE execution failed"),
                    );
                });
            });
            group.bench_function("native_resident_inputs", |bencher| {
                bencher.iter(|| {
                    black_box(
                        native
                            .execute_resident(&native_prediction, &native_target, count)
                            .expect("resident native MSE execution failed"),
                    );
                });
            });
            group.finish();
        }
        verify(
            reference.data(),
            assessor
                .execute_graph(&graph, &inputs, loss, selected.pool, &mut memory)?
                .data(),
        )?;
        verify(reference.data(), &[native.execute(&prediction, &target)?])?;
        let resident_after = assessor.execute_prepared_with_resources(
            &prepared,
            &resident_inputs,
            selected.pool,
            &mut memory,
        )?;
        verify(reference.data(), resident_after.data())?;
    }
    Ok(())
}

fn verify_reduction_contract(runtime: &HipRuntime, blas: &Rocblas) -> Result<(), Box<dyn Error>> {
    let fixtures = [
        (
            "finite nonnegative",
            (0..1_048_576)
                .map(|index| match index % 8 {
                    0 => 0.0_f32,
                    1 => 0.25,
                    2 | 6 => 1.0,
                    3 => f32::MIN_POSITIVE,
                    4 => 3.0,
                    5 => 1.0e20,
                    _ => 2.0,
                })
                .collect(),
            0,
        ),
        ("signed zero", vec![-0.0_f32, 0.0, -0.0, 0.0], 1),
        ("infinity", vec![0.0_f32, f32::INFINITY, 2.0], 2),
        ("NaN", vec![0.0_f32, f32::NAN, f32::INFINITY], 3),
    ];
    for (name, values, expected_kind) in fixtures {
        let count = values.len();
        let mut input = runtime.allocate(count * size_of::<f32>())?;
        input.copy_from(bytemuck::cast_slice(&values))?;
        let mut ones = runtime.allocate(count * size_of::<f32>())?;
        ones.copy_from(bytemuck::cast_slice(&vec![1.0_f32; count]))?;
        let dot = runtime.allocate(size_of::<f32>())?;
        let asum = runtime.allocate(size_of::<f32>())?;
        blas.sdot_scaled(count, &input, 1, &ones, 1, 1.0, &dot)?;
        blas.sasum_scaled(count, &input, 1, 1.0, &asum)?;
        let mut dot_value = 0.0_f32;
        let mut asum_value = 0.0_f32;
        dot.copy_to(bytemuck::bytes_of_mut(&mut dot_value))?;
        asum.copy_to(bytemuck::bytes_of_mut(&mut asum_value))?;
        let agrees = match expected_kind {
            0 => dot_value.to_bits() == asum_value.to_bits(),
            1 => {
                dot_value == 0.0
                    && asum_value == 0.0
                    && !dot_value.is_sign_negative()
                    && !asum_value.is_sign_negative()
            }
            2 => dot_value.is_infinite() && asum_value.is_infinite(),
            3 => dot_value.is_nan() && asum_value.is_nan(),
            _ => unreachable!(),
        };
        if !agrees {
            return Err(format!(
                "rocBLAS sasum is not a safe MSE reduction replacement for {name}: sdot={dot_value:?}, sasum={asum_value:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn verify(expected: &[f32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    if expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| (expected - actual).abs() <= 1.0e-5)
    {
        Ok(())
    } else {
        Err(format!("MSE output mismatch: expected {expected:?}, got {actual:?}").into())
    }
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = bench
}
criterion_main!(benches);

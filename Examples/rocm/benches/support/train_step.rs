//! Direct HIP and rocBLAS peer for the linear-regression training graph.

use std::error::Error;

use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    Rocblas,
    RocmDiscovery,
    compile_hip_source_for_device,
};
use fusion_pcu::PcuObjectRef;

#[derive(Clone, Copy)]
pub struct TrainInputs<'a> {
    pub rows: usize,
    pub features: usize,
    pub samples: &'a [f32],
    pub target: &'a [f32],
    pub factor: &'a [f32],
    pub initial_weights: &'a [f32],
    pub rate: &'a [f32],
}

const SOURCE: &str = r#"
extern "C" __global__ void training_error(
    const float *prediction, const float *target, const float *factor,
    float *error, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) error[id] = (prediction[id] - target[id]) * factor[id];
}
extern "C" __global__ void training_update(
    const float *weights, const float *gradient, const float *rate,
    float *updated, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) updated[id] = weights[id] - rate[id] * gradient[id];
}
"#;

/// Preallocated direct HIP/rocBLAS route; all graph inputs stay device resident.
pub struct NativeTrainStep {
    stream: HipStreamHandle,
    blas: Rocblas,
    error_kernel: HipKernel,
    update_kernel: HipKernel,
    samples: DeviceBuffer,
    target: DeviceBuffer,
    factor: DeviceBuffer,
    rate: DeviceBuffer,
    prediction: DeviceBuffer,
    error: DeviceBuffer,
    gradient: DeviceBuffer,
    updated: DeviceBuffer,
    initial_weights: DeviceBuffer,
    weights: DeviceBuffer,
    rows: usize,
    features: usize,
}

impl NativeTrainStep {
    pub fn prepare(
        discovery: &RocmDiscovery,
        device: PcuObjectRef,
        input: &TrainInputs<'_>,
    ) -> Result<Self, Box<dyn Error>> {
        let TrainInputs {
            rows,
            features,
            samples,
            target,
            factor,
            initial_weights,
            rate,
        } = *input;
        let runtime = discovery.open_device(device)?;
        let image = compile_hip_source_for_device(&runtime, SOURCE)?;
        let module = runtime.load_module(&image)?;
        let error_kernel = module.function(c"training_error")?;
        let update_kernel = module.function(c"training_update")?;
        let stream = runtime.create_stream()?;
        let blas = Rocblas::new(&runtime)?;
        let samples_buffer = upload(&runtime, samples)?;
        let target_buffer = upload(&runtime, target)?;
        let factor_buffer = upload(&runtime, factor)?;
        let rate_buffer = upload(&runtime, rate)?;
        let prediction = allocate(&runtime, rows)?;
        let error = allocate(&runtime, rows)?;
        let gradient = allocate(&runtime, features)?;
        let updated = allocate(&runtime, features)?;
        // Keep the caller's starting point immutable and resident. Each execution writes its
        // two updates into separate reusable buffers without resetting the initial allocation.
        let initial_weights = upload(&runtime, initial_weights)?;
        let weights = allocate(&runtime, features)?;
        Ok(Self {
            stream,
            blas,
            error_kernel,
            update_kernel,
            samples: samples_buffer,
            target: target_buffer,
            factor: factor_buffer,
            rate: rate_buffer,
            prediction,
            error,
            gradient,
            updated,
            initial_weights,
            weights,
            rows,
            features,
        })
    }

    pub fn execute_two(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        for (weights, updated) in [
            (&self.initial_weights, &self.weights),
            (&self.weights, &self.updated),
        ] {
            // Row-major X*W is column-major W^T*X^T.
            self.blas.sgemm(
                false,
                false,
                1,
                self.rows,
                self.features,
                1.0,
                weights,
                1,
                &self.samples,
                self.features,
                0.0,
                &self.prediction,
                1,
            )?;
            launch_error(
                &self.error_kernel,
                &self.stream,
                &self.prediction,
                &self.target,
                &self.factor,
                &self.error,
                self.rows,
            )?;
            // Compute the row-major X^T * error through its column-major transpose:
            // error^T * X. This matches the ROCm tensor adapter's rocBLAS orientation.
            self.blas.sgemm(
                false,
                true,
                1,
                self.features,
                self.rows,
                1.0,
                &self.error,
                1,
                &self.samples,
                self.features,
                0.0,
                &self.gradient,
                1,
            )?;
            launch_update(
                &self.update_kernel,
                &self.stream,
                weights,
                &self.gradient,
                &self.rate,
                updated,
                self.features,
            )?;
        }
        let mut result = vec![0.0_f32; self.features];
        self.updated
            .copy_to(bytemuck::cast_slice_mut(&mut result))?;
        Ok(result)
    }
}

fn allocate(runtime: &HipRuntime, elements: usize) -> Result<DeviceBuffer, Box<dyn Error>> {
    Ok(runtime.allocate(
        elements
            .checked_mul(size_of::<f32>())
            .ok_or("buffer size overflow")?,
    )?)
}

fn upload(runtime: &HipRuntime, values: &[f32]) -> Result<DeviceBuffer, Box<dyn Error>> {
    let mut buffer = allocate(runtime, values.len())?;
    buffer.copy_from(bytemuck::cast_slice(values))?;
    Ok(buffer)
}

fn launch_error(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    prediction: &DeviceBuffer,
    target: &DeviceBuffer,
    factor: &DeviceBuffer,
    error: &DeviceBuffer,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    launch(kernel, stream, prediction, target, factor, error, count)
}

fn launch_update(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    weights: &DeviceBuffer,
    gradient: &DeviceBuffer,
    rate: &DeviceBuffer,
    updated: &DeviceBuffer,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    launch(kernel, stream, weights, gradient, rate, updated, count)
}

fn launch(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
    output: &DeviceBuffer,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    let count_u32 = u32::try_from(count)?;
    let count_bytes = count_u32.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(a),
        HipKernelArgument::Buffer(b),
        HipKernelArgument::Buffer(c),
        HipKernelArgument::Buffer(output),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    // SAFETY: The kernels take four f32 pointers then a u32 length. All buffers are large enough,
    // the grid-stride tail is guarded, and waiting establishes lifetime-safe completion.
    #[allow(unsafe_code)]
    let mut completion = unsafe {
        kernel.launch(
            stream,
            [count_u32.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &args,
        )?
    };
    completion.wait()?;
    Ok(())
}

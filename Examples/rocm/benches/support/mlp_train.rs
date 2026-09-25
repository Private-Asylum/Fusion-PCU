//! Direct HIP and rocBLAS peer for two-step training of a 1024-2048-2048-1024 MLP.

use std::error::Error;
use std::time::{
    Duration,
    Instant,
};

use fusion_pcu::PcuObjectRef;
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

const INPUTS: usize = 1024;
const HIDDEN: usize = 2048;
const OUTPUTS: usize = 1024;

#[derive(Clone, Copy)]
pub struct TrainInputs<'a> {
    pub batch: usize,
    pub samples: &'a [f32],
    pub targets: &'a [f32],
    pub initial_w1: &'a [f32],
    pub initial_w2: &'a [f32],
    pub initial_w3: &'a [f32],
    pub learning_rate: f32,
}

pub struct MlpWeights {
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
    /// MSE before each of the two corresponding SGD updates.
    pub losses: [f32; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeMlpProfile {
    pub forward: [Duration; 2],
    pub backward: [Duration; 2],
    pub update: [Duration; 2],
    pub reset_weights: Duration,
    pub allocation_count: usize,
    pub allocated_bytes: u64,
    pub allocation_time: Duration,
    pub upload_count: usize,
    pub uploaded_bytes: usize,
    pub upload_time: Duration,
    pub download_count: usize,
    pub downloaded_bytes: usize,
    pub download_time: Duration,
    /// Sum of live HIP buffers owned by this prepared route.
    pub peak_device_bytes: u64,
}

#[derive(Default)]
struct MemoryStats {
    profile: NativeMlpProfile,
}

const SOURCE: &str = r#"
extern "C" __global__ void relu_forward(
    const float *input, float *output, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = fmaxf(input[id], 0.0f);
}
extern "C" __global__ void relu_backward(
    const float *input, const float *upstream, float *output, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = input[id] > 0.0f ? upstream[id] : 0.0f;
}
extern "C" __global__ void mse_gradient(
    const float *prediction, const float *target, float *gradient,
    unsigned int n, float scale) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) gradient[id] = (prediction[id] - target[id]) * scale;
}
extern "C" __global__ void squared_difference(
    const float *prediction, const float *target, float *squares,
    unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float difference = prediction[id] - target[id];
        squares[id] = difference * difference;
    }
}
extern "C" __global__ void sgd_update(
    const float *weights, const float *gradient, float *updated,
    unsigned int n, float learning_rate) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) updated[id] = weights[id] - learning_rate * gradient[id];
}
"#;

/// Preallocated native route. Inputs, starting weights, and all intermediates remain resident.
pub struct NativeMlpTrain {
    stream: HipStreamHandle,
    blas: Rocblas,
    relu_forward: HipKernel,
    relu_backward: HipKernel,
    mse_gradient: HipKernel,
    squared_difference: HipKernel,
    sgd_update: HipKernel,
    samples: DeviceBuffer,
    targets: DeviceBuffer,
    initial_w1: DeviceBuffer,
    initial_w2: DeviceBuffer,
    initial_w3: DeviceBuffer,
    weights_a1: DeviceBuffer,
    weights_a2: DeviceBuffer,
    weights_a3: DeviceBuffer,
    weights_b1: DeviceBuffer,
    weights_b2: DeviceBuffer,
    weights_b3: DeviceBuffer,
    z1: DeviceBuffer,
    a1: DeviceBuffer,
    z2: DeviceBuffer,
    a2: DeviceBuffer,
    prediction: DeviceBuffer,
    squared_differences: DeviceBuffer,
    reduction_ones: DeviceBuffer,
    prediction_gradient: DeviceBuffer,
    gradient_w1: DeviceBuffer,
    gradient_w2: DeviceBuffer,
    gradient_w3: DeviceBuffer,
    activation_gradient_1: DeviceBuffer,
    activation_gradient_2: DeviceBuffer,
    preactivation_gradient_1: DeviceBuffer,
    preactivation_gradient_2: DeviceBuffer,
    losses: [DeviceBuffer; 2],
    batch: usize,
    learning_rate: f32,
    preparation_profile: NativeMlpProfile,
}

impl NativeMlpTrain {
    #[allow(clippy::similar_names)] // A/B name the two alternating resident weight banks.
    pub fn prepare(
        discovery: &RocmDiscovery,
        device: PcuObjectRef,
        inputs: &TrainInputs<'_>,
    ) -> Result<Self, Box<dyn Error>> {
        validate_inputs(inputs)?;
        let runtime = discovery.open_device(device)?;
        let image = compile_hip_source_for_device(&runtime, SOURCE)?;
        let module = runtime.load_module(&image)?;
        let relu_forward = module.function(c"relu_forward")?;
        let relu_backward = module.function(c"relu_backward")?;
        let mse_gradient = module.function(c"mse_gradient")?;
        let squared_difference = module.function(c"squared_difference")?;
        let sgd_update = module.function(c"sgd_update")?;
        let stream = runtime.create_stream()?;
        let blas = Rocblas::new(&runtime)?;

        let mut stats = MemoryStats::default();
        let samples = upload(&runtime, inputs.samples, &mut stats)?;
        let targets = upload(&runtime, inputs.targets, &mut stats)?;
        let initial_w1 = upload(&runtime, inputs.initial_w1, &mut stats)?;
        let initial_w2 = upload(&runtime, inputs.initial_w2, &mut stats)?;
        let initial_w3 = upload(&runtime, inputs.initial_w3, &mut stats)?;
        let weights_a1 = allocate(&runtime, INPUTS * HIDDEN, &mut stats)?;
        let weights_a2 = allocate(&runtime, HIDDEN * HIDDEN, &mut stats)?;
        let weights_a3 = allocate(&runtime, HIDDEN * OUTPUTS, &mut stats)?;
        let weights_b1 = allocate(&runtime, INPUTS * HIDDEN, &mut stats)?;
        let weights_b2 = allocate(&runtime, HIDDEN * HIDDEN, &mut stats)?;
        let weights_b3 = allocate(&runtime, HIDDEN * OUTPUTS, &mut stats)?;

        let batch_hidden = inputs
            .batch
            .checked_mul(HIDDEN)
            .ok_or("activation extent overflow")?;
        let batch_output = inputs
            .batch
            .checked_mul(OUTPUTS)
            .ok_or("output extent overflow")?;
        let z1 = allocate(&runtime, batch_hidden, &mut stats)?;
        let a1 = allocate(&runtime, batch_hidden, &mut stats)?;
        let z2 = allocate(&runtime, batch_hidden, &mut stats)?;
        let a2 = allocate(&runtime, batch_hidden, &mut stats)?;
        let prediction = allocate(&runtime, batch_output, &mut stats)?;
        let squared_differences = allocate(&runtime, batch_output, &mut stats)?;
        let reduction_ones = upload(&runtime, &vec![1.0_f32; batch_output], &mut stats)?;
        let prediction_gradient = allocate(&runtime, batch_output, &mut stats)?;
        let gradient_w1 = allocate(&runtime, INPUTS * HIDDEN, &mut stats)?;
        let gradient_w2 = allocate(&runtime, HIDDEN * HIDDEN, &mut stats)?;
        let gradient_w3 = allocate(&runtime, HIDDEN * OUTPUTS, &mut stats)?;
        let activation_gradient_1 = allocate(&runtime, batch_hidden, &mut stats)?;
        let activation_gradient_2 = allocate(&runtime, batch_hidden, &mut stats)?;
        let preactivation_gradient_1 = allocate(&runtime, batch_hidden, &mut stats)?;
        let preactivation_gradient_2 = allocate(&runtime, batch_hidden, &mut stats)?;
        let losses = [
            allocate(&runtime, 1, &mut stats)?,
            allocate(&runtime, 1, &mut stats)?,
        ];

        Ok(Self {
            stream,
            blas,
            relu_forward,
            relu_backward,
            mse_gradient,
            squared_difference,
            sgd_update,
            samples,
            targets,
            initial_w1,
            initial_w2,
            initial_w3,
            weights_a1,
            weights_a2,
            weights_a3,
            weights_b1,
            weights_b2,
            weights_b3,
            z1,
            a1,
            z2,
            a2,
            prediction,
            squared_differences,
            reduction_ones,
            prediction_gradient,
            gradient_w1,
            gradient_w2,
            gradient_w3,
            activation_gradient_1,
            activation_gradient_2,
            preactivation_gradient_1,
            preactivation_gradient_2,
            losses,
            batch: inputs.batch,
            learning_rate: inputs.learning_rate,
            preparation_profile: stats.profile,
        })
    }

    /// Restores the immutable starting point device-to-device and performs two forward/backward
    /// and SGD steps. Only the final weights and the two scalar losses are read back.
    pub fn execute_two_steps(&mut self) -> Result<MlpWeights, Box<dyn Error>> {
        self.reset_weights()?;
        self.execute_step(
            &self.weights_a1,
            &self.weights_a2,
            &self.weights_a3,
            &self.weights_b1,
            &self.weights_b2,
            &self.weights_b3,
            &self.losses[0],
        )?;
        self.execute_step(
            &self.weights_b1,
            &self.weights_b2,
            &self.weights_b3,
            &self.weights_a1,
            &self.weights_a2,
            &self.weights_a3,
            &self.losses[1],
        )?;

        Ok(MlpWeights {
            w1: download(&self.weights_a1, INPUTS * HIDDEN)?,
            w2: download(&self.weights_a2, HIDDEN * HIDDEN)?,
            w3: download(&self.weights_a3, HIDDEN * OUTPUTS)?,
            losses: [
                download(&self.losses[0], 1)?[0],
                download(&self.losses[1], 1)?[0],
            ],
        })
    }

    /// Runs a diagnostic pass with host timings around synchronous native stage groups.
    /// The regular benchmark path uses `execute_two_steps` and carries no timer overhead.
    pub fn execute_two_steps_profiled(
        &mut self,
    ) -> Result<(MlpWeights, NativeMlpProfile), Box<dyn Error>> {
        let mut profile = self.preparation_profile;
        let start = Instant::now();
        self.reset_weights()?;
        profile.reset_weights = start.elapsed();
        let first = self.execute_step_profiled(
            &self.weights_a1,
            &self.weights_a2,
            &self.weights_a3,
            &self.weights_b1,
            &self.weights_b2,
            &self.weights_b3,
            &self.losses[0],
        )?;
        let second = self.execute_step_profiled(
            &self.weights_b1,
            &self.weights_b2,
            &self.weights_b3,
            &self.weights_a1,
            &self.weights_a2,
            &self.weights_a3,
            &self.losses[1],
        )?;
        profile.forward = [first.0, second.0];
        profile.backward = [first.1, second.1];
        profile.update = [first.2, second.2];
        let start = Instant::now();
        let weights = self.download_weights()?;
        profile.download_time += start.elapsed();
        profile.download_count += 5;
        profile.downloaded_bytes +=
            (INPUTS * HIDDEN * 2 + HIDDEN * HIDDEN) * size_of::<f32>() + 2 * size_of::<f32>();
        Ok((weights, profile))
    }

    fn download_weights(&self) -> Result<MlpWeights, Box<dyn Error>> {
        Ok(MlpWeights {
            w1: download(&self.weights_a1, INPUTS * HIDDEN)?,
            w2: download(&self.weights_a2, HIDDEN * HIDDEN)?,
            w3: download(&self.weights_a3, HIDDEN * OUTPUTS)?,
            losses: [
                download(&self.losses[0], 1)?[0],
                download(&self.losses[1], 1)?[0],
            ],
        })
    }

    fn reset_weights(&mut self) -> Result<(), Box<dyn Error>> {
        self.weights_a1
            .copy_from_device(&self.initial_w1, self.initial_w1.len())?;
        self.weights_a2
            .copy_from_device(&self.initial_w2, self.initial_w2.len())?;
        self.weights_a3
            .copy_from_device(&self.initial_w3, self.initial_w3.len())?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn execute_step(
        &self,
        w1: &DeviceBuffer,
        w2: &DeviceBuffer,
        w3: &DeviceBuffer,
        updated_w1: &DeviceBuffer,
        updated_w2: &DeviceBuffer,
        updated_w3: &DeviceBuffer,
        loss: &DeviceBuffer,
    ) -> Result<(), Box<dyn Error>> {
        self.execute_step_inner(w1, w2, w3, updated_w1, updated_w2, updated_w3, loss, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_step_profiled(
        &self,
        w1: &DeviceBuffer,
        w2: &DeviceBuffer,
        w3: &DeviceBuffer,
        updated_w1: &DeviceBuffer,
        updated_w2: &DeviceBuffer,
        updated_w3: &DeviceBuffer,
        loss: &DeviceBuffer,
    ) -> Result<(Duration, Duration, Duration), Box<dyn Error>> {
        let mut times = [Duration::ZERO; 3];
        self.execute_step_inner(
            w1,
            w2,
            w3,
            updated_w1,
            updated_w2,
            updated_w3,
            loss,
            Some(&mut times),
        )?;
        Ok(times.into())
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn execute_step_inner(
        &self,
        w1: &DeviceBuffer,
        w2: &DeviceBuffer,
        w3: &DeviceBuffer,
        updated_w1: &DeviceBuffer,
        updated_w2: &DeviceBuffer,
        updated_w3: &DeviceBuffer,
        loss: &DeviceBuffer,
        mut timings: Option<&mut [Duration; 3]>,
    ) -> Result<(), Box<dyn Error>> {
        let batch = self.batch;
        let batch_output = batch.checked_mul(OUTPUTS).ok_or("output extent overflow")?;
        let batch_hidden = batch
            .checked_mul(HIDDEN)
            .ok_or("activation extent overflow")?;
        let output_u32 = u32::try_from(batch_output)?;
        let hidden_u32 = u32::try_from(batch_hidden)?;
        // f32 represents every integer exactly through 2^24, so the MSE divisor and
        // derivative scale do not silently round the element count.
        if batch_output > (1 << 24) {
            return Err("batch output extent exceeds exact f32 integer range".into());
        }
        #[allow(clippy::cast_precision_loss)] // Validated as exactly representable in f32.
        let mse_scale = 2.0_f32 / batch_output as f32;

        let mut stage_start = timings.as_ref().map(|_| Instant::now());
        matmul(
            &self.blas,
            &self.samples,
            w1,
            &self.z1,
            batch,
            INPUTS,
            HIDDEN,
        )?;
        launch_relu(
            &self.relu_forward,
            &self.stream,
            &self.z1,
            &self.a1,
            hidden_u32,
        )?;
        matmul(&self.blas, &self.a1, w2, &self.z2, batch, HIDDEN, HIDDEN)?;
        launch_relu(
            &self.relu_forward,
            &self.stream,
            &self.z2,
            &self.a2,
            hidden_u32,
        )?;
        matmul(
            &self.blas,
            &self.a2,
            w3,
            &self.prediction,
            batch,
            HIDDEN,
            OUTPUTS,
        )?;
        launch_squared_difference(
            &self.squared_difference,
            &self.stream,
            &self.prediction,
            &self.targets,
            &self.squared_differences,
            output_u32,
        )?;
        // Match the PCU reduction route: squared differences dotted with a resident ones vector.
        self.blas.sdot_scaled(
            batch_output,
            &self.squared_differences,
            1,
            &self.reduction_ones,
            1,
            mse_scale * 0.5,
            loss,
        )?;
        launch_mse_gradient(
            &self.mse_gradient,
            &self.stream,
            &self.prediction,
            &self.targets,
            &self.prediction_gradient,
            output_u32,
            mse_scale,
        )?;
        if let (Some(timings), Some(start)) = (timings.as_deref_mut(), stage_start.take()) {
            timings[0] += start.elapsed();
        }
        stage_start = timings.as_ref().map(|_| Instant::now());

        // dW3 = A2^T * dY, using the row-major-to-column-major transpose identity.
        matmul_left_transpose(
            &self.blas,
            &self.a2,
            &self.prediction_gradient,
            &self.gradient_w3,
            batch,
            HIDDEN,
            OUTPUTS,
        )?;
        // dA2 = dY * W3^T.
        matmul_right_transpose(
            &self.blas,
            &self.prediction_gradient,
            w3,
            &self.activation_gradient_2,
            batch,
            OUTPUTS,
            HIDDEN,
        )?;
        launch_relu_backward(
            &self.relu_backward,
            &self.stream,
            &self.z2,
            &self.activation_gradient_2,
            &self.preactivation_gradient_2,
            hidden_u32,
        )?;
        // dW2 = A1^T * dZ2.
        matmul_left_transpose(
            &self.blas,
            &self.a1,
            &self.preactivation_gradient_2,
            &self.gradient_w2,
            batch,
            HIDDEN,
            HIDDEN,
        )?;
        // dA1 = dZ2 * W2^T.
        matmul_right_transpose(
            &self.blas,
            &self.preactivation_gradient_2,
            w2,
            &self.activation_gradient_1,
            batch,
            HIDDEN,
            HIDDEN,
        )?;
        launch_relu_backward(
            &self.relu_backward,
            &self.stream,
            &self.z1,
            &self.activation_gradient_1,
            &self.preactivation_gradient_1,
            hidden_u32,
        )?;
        // dW1 = X^T * dZ1.
        matmul_left_transpose(
            &self.blas,
            &self.samples,
            &self.preactivation_gradient_1,
            &self.gradient_w1,
            batch,
            INPUTS,
            HIDDEN,
        )?;
        if let (Some(timings), Some(start)) = (timings.as_deref_mut(), stage_start.take()) {
            timings[1] += start.elapsed();
        }
        stage_start = timings.as_ref().map(|_| Instant::now());

        launch_sgd(
            &self.sgd_update,
            &self.stream,
            w1,
            &self.gradient_w1,
            updated_w1,
            u32::try_from(INPUTS * HIDDEN)?,
            self.learning_rate,
        )?;
        launch_sgd(
            &self.sgd_update,
            &self.stream,
            w2,
            &self.gradient_w2,
            updated_w2,
            u32::try_from(HIDDEN * HIDDEN)?,
            self.learning_rate,
        )?;
        launch_sgd(
            &self.sgd_update,
            &self.stream,
            w3,
            &self.gradient_w3,
            updated_w3,
            u32::try_from(HIDDEN * OUTPUTS)?,
            self.learning_rate,
        )?;
        if let (Some(timings), Some(start)) = (timings, stage_start) {
            timings[2] += start.elapsed();
        }
        Ok(())
    }
}

fn validate_inputs(inputs: &TrainInputs<'_>) -> Result<(), Box<dyn Error>> {
    if inputs.batch == 0 || !inputs.learning_rate.is_finite() {
        return Err("batch must be nonzero and learning rate finite".into());
    }
    let expected = [
        inputs
            .batch
            .checked_mul(INPUTS)
            .ok_or("input extent overflow")?,
        inputs
            .batch
            .checked_mul(OUTPUTS)
            .ok_or("target extent overflow")?,
        INPUTS * HIDDEN,
        HIDDEN * HIDDEN,
        HIDDEN * OUTPUTS,
    ];
    let actual = [
        inputs.samples.len(),
        inputs.targets.len(),
        inputs.initial_w1.len(),
        inputs.initial_w2.len(),
        inputs.initial_w3.len(),
    ];
    if actual != expected {
        return Err(format!("input lengths {actual:?} do not match {expected:?}").into());
    }
    u32::try_from(
        inputs
            .batch
            .checked_mul(OUTPUTS)
            .ok_or("output extent overflow")?,
    )?;
    u32::try_from(
        inputs
            .batch
            .checked_mul(HIDDEN)
            .ok_or("activation extent overflow")?,
    )?;
    i32::try_from(inputs.batch)?;
    if inputs.batch * OUTPUTS > (1 << 24) {
        return Err("batch output extent exceeds exact f32 integer range".into());
    }
    Ok(())
}

fn allocate(
    runtime: &HipRuntime,
    elements: usize,
    stats: &mut MemoryStats,
) -> Result<DeviceBuffer, Box<dyn Error>> {
    let bytes = elements
        .checked_mul(size_of::<f32>())
        .ok_or("buffer byte size overflow")?;
    let start = Instant::now();
    let result = runtime.allocate(bytes)?;
    stats.profile.allocation_time += start.elapsed();
    stats.profile.allocation_count += 1;
    stats.profile.allocated_bytes += u64::try_from(bytes)?;
    stats.profile.peak_device_bytes += u64::try_from(bytes)?;
    Ok(result)
}

fn upload(
    runtime: &HipRuntime,
    values: &[f32],
    stats: &mut MemoryStats,
) -> Result<DeviceBuffer, Box<dyn Error>> {
    let mut buffer = allocate(runtime, values.len(), stats)?;
    let bytes = bytemuck::cast_slice(values);
    let start = Instant::now();
    buffer.copy_from(bytes)?;
    stats.profile.upload_time += start.elapsed();
    stats.profile.upload_count += 1;
    stats.profile.uploaded_bytes += bytes.len();
    Ok(buffer)
}

fn download(buffer: &DeviceBuffer, elements: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut values = vec![0.0_f32; elements];
    buffer.copy_to(bytemuck::cast_slice_mut(&mut values))?;
    Ok(values)
}

fn matmul(
    blas: &Rocblas,
    left: &DeviceBuffer,
    right: &DeviceBuffer,
    output: &DeviceBuffer,
    rows: usize,
    inner: usize,
    columns: usize,
) -> Result<(), Box<dyn Error>> {
    // Row-major A*B is column-major B^T*A^T.
    blas.sgemm(
        false, false, columns, rows, inner, 1.0, right, columns, left, inner, 0.0, output, columns,
    )?;
    Ok(())
}

fn matmul_left_transpose(
    blas: &Rocblas,
    left: &DeviceBuffer,
    right: &DeviceBuffer,
    output: &DeviceBuffer,
    reduction: usize,
    rows: usize,
    columns: usize,
) -> Result<(), Box<dyn Error>> {
    // Row-major left^T*right is column-major right^T*left. The right-hand column
    // representation therefore uses a transpose flag.
    blas.sgemm(
        false, true, columns, rows, reduction, 1.0, right, columns, left, rows, 0.0, output,
        columns,
    )?;
    Ok(())
}

fn matmul_right_transpose(
    blas: &Rocblas,
    left: &DeviceBuffer,
    right: &DeviceBuffer,
    output: &DeviceBuffer,
    rows: usize,
    inner: usize,
    columns: usize,
) -> Result<(), Box<dyn Error>> {
    // Row-major left*right^T is column-major right*left^T.
    blas.sgemm(
        true, false, columns, rows, inner, 1.0, right, inner, left, inner, 0.0, output, columns,
    )?;
    Ok(())
}

fn launch_relu(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    input: &DeviceBuffer,
    output: &DeviceBuffer,
    count: u32,
) -> Result<(), Box<dyn Error>> {
    let count_bytes = count.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(input),
        HipKernelArgument::Buffer(output),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    launch(kernel, stream, &args, count)
}

fn launch_relu_backward(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    input: &DeviceBuffer,
    upstream: &DeviceBuffer,
    output: &DeviceBuffer,
    count: u32,
) -> Result<(), Box<dyn Error>> {
    let count_bytes = count.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(input),
        HipKernelArgument::Buffer(upstream),
        HipKernelArgument::Buffer(output),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    launch(kernel, stream, &args, count)
}

fn launch_mse_gradient(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    prediction: &DeviceBuffer,
    target: &DeviceBuffer,
    gradient: &DeviceBuffer,
    count: u32,
    scale: f32,
) -> Result<(), Box<dyn Error>> {
    let count_bytes = count.to_ne_bytes();
    let scale_bytes = scale.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(prediction),
        HipKernelArgument::Buffer(target),
        HipKernelArgument::Buffer(gradient),
        HipKernelArgument::Bytes(&count_bytes),
        HipKernelArgument::Bytes(&scale_bytes),
    ];
    launch(kernel, stream, &args, count)
}

fn launch_squared_difference(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    prediction: &DeviceBuffer,
    target: &DeviceBuffer,
    squares: &DeviceBuffer,
    count: u32,
) -> Result<(), Box<dyn Error>> {
    let count_bytes = count.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(prediction),
        HipKernelArgument::Buffer(target),
        HipKernelArgument::Buffer(squares),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    launch(kernel, stream, &args, count)
}

fn launch_sgd(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    weights: &DeviceBuffer,
    gradient: &DeviceBuffer,
    updated: &DeviceBuffer,
    count: u32,
    learning_rate: f32,
) -> Result<(), Box<dyn Error>> {
    let count_bytes = count.to_ne_bytes();
    let rate_bytes = learning_rate.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(weights),
        HipKernelArgument::Buffer(gradient),
        HipKernelArgument::Buffer(updated),
        HipKernelArgument::Bytes(&count_bytes),
        HipKernelArgument::Bytes(&rate_bytes),
    ];
    launch(kernel, stream, &args, count)
}

fn launch(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    args: &[HipKernelArgument<'_>],
    count: u32,
) -> Result<(), Box<dyn Error>> {
    let blocks = count.div_ceil(256);
    // SAFETY: Every call site matches the declared pointer/scalar argument order and allocates
    // enough elements for `count`. Padded lanes are guarded; waiting keeps all buffers alive.
    #[allow(unsafe_code)]
    let mut completion = unsafe { kernel.launch(stream, [blocks, 1, 1], [256, 1, 1], 0, args)? };
    completion.wait()?;
    Ok(())
}

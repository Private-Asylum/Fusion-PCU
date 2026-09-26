//! Preallocated direct HIP references for strict and explicitly fused SGD arithmetic.

use std::error::Error;

use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    RocmDiscovery,
    compile_hip_source_for_device,
};
use fusion_pcu::PcuObjectRef;

const SOURCE: &str = r#"
extern "C" __global__ void sgd_strict(
    const float *weights, const float *gradient, float *output,
    float learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        volatile float product = learning_rate * gradient[id];
        output[id] = weights[id] - product;
    }
}
extern "C" __global__ void sgd_fmaf(
    const float *weights, const float *gradient, float *output,
    float learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = __builtin_fmaf(-learning_rate, gradient[id], weights[id]);
}
"#;

/// Native HIP kernels and resident storage, compiled and allocated once before timing.
pub struct NativeSgd {
    stream: HipStreamHandle,
    strict: HipKernel,
    fused: HipKernel,
    weights: DeviceBuffer,
    gradient: DeviceBuffer,
    output: DeviceBuffer,
    elements: u32,
    rate: f32,
    _runtime: HipRuntime,
}

impl NativeSgd {
    pub fn prepare(
        discovery: &RocmDiscovery,
        device: PcuObjectRef,
        weights: &[f32],
        gradient: &[f32],
        rate: f32,
    ) -> Result<Self, Box<dyn Error>> {
        if weights.is_empty() || weights.len() != gradient.len() {
            return Err("native SGD input shapes must be equal and nonempty".into());
        }
        let elements = u32::try_from(weights.len())?;
        let runtime = discovery.open_device(device)?;
        let image = compile_hip_source_for_device(&runtime, SOURCE)?;
        let module = runtime.load_module(&image)?;
        let strict = module.function(c"sgd_strict")?;
        let fused = module.function(c"sgd_fmaf")?;
        let stream = runtime.create_stream()?;
        let mut weight_buffer = runtime.allocate(std::mem::size_of_val(weights))?;
        let mut gradient_buffer = runtime.allocate(std::mem::size_of_val(gradient))?;
        let output = runtime.allocate(std::mem::size_of_val(weights))?;
        weight_buffer.copy_from(bytemuck::cast_slice(weights))?;
        gradient_buffer.copy_from(bytemuck::cast_slice(gradient))?;
        Ok(Self {
            stream,
            strict,
            fused,
            weights: weight_buffer,
            gradient: gradient_buffer,
            output,
            elements,
            rate,
            _runtime: runtime,
        })
    }

    pub fn execute_resident(&self, contracted: bool) -> Result<(), Box<dyn Error>> {
        let kernel = if contracted {
            &self.fused
        } else {
            &self.strict
        };
        let rate_bytes = self.rate.to_ne_bytes();
        let count_bytes = self.elements.to_ne_bytes();
        let arguments = [
            HipKernelArgument::Buffer(&self.weights),
            HipKernelArgument::Buffer(&self.gradient),
            HipKernelArgument::Buffer(&self.output),
            HipKernelArgument::Bytes(&rate_bytes),
            HipKernelArgument::Bytes(&count_bytes),
        ];
        // SAFETY: Each pointer references an allocation sized for `elements` f32 values; one
        // invocation writes each in-bounds element. The completion is awaited before readback.
        #[allow(unsafe_code)]
        let mut completion = unsafe {
            kernel.launch(
                &self.stream,
                [self.elements.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &arguments,
            )?
        };
        completion.wait()?;
        Ok(())
    }

    /// Copy the last completed output to a newly allocated host buffer.
    ///
    /// The HIP copy is synchronous, so this isolates device-to-host transfer from dispatch and
    /// completion while retaining the same host allocation behavior as `execute`.
    pub fn readback(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut values = vec![0.0_f32; self.elements as usize];
        self.output.copy_to(bytemuck::cast_slice_mut(&mut values))?;
        Ok(values)
    }

    pub fn execute(&self, contracted: bool) -> Result<Vec<f32>, Box<dyn Error>> {
        self.execute_resident(contracted)?;
        self.readback()
    }
}

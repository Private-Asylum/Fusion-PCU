//! Matched native SGD: immutable resident inputs, fresh output, one launch and terminal event wait.
#[rustfmt::skip]
use std::{
    error::Error,
    mem::{
        size_of,
    },
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuHostArgument,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    compile_hip_source,
};
const SOURCE: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void sgd_preserved(const float *weights, const float *gradient,
    float *output, float rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) { volatile float product = rate * gradient[id]; output[id] = weights[id] - product; }
}
extern "C" __global__ void sgd_contracted(const float *weights, const float *gradient,
    float *output, float rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = __builtin_fmaf(-rate, gradient[id], weights[id]);
}
"#;
pub struct Native {
    runtime: HipRuntime,
    stream: HipStreamHandle,
    preserved: HipKernel,
    contracted: HipKernel,
    weights: [DeviceBuffer; 3],
    gradient: [DeviceBuffer; 3],
    count: u32,
}
impl Native {
    pub fn new<const N: usize>(runtime: &HipRuntime) -> Result<Self, Box<dyn Error>> {
        let architecture = runtime
            .device_info()?
            .architecture
            .ok_or("selected device lacks architecture")?;
        let image = compile_hip_source(SOURCE, &architecture)?;
        let module = runtime.load_module(&image)?;
        let preserved = module.function(c"sgd_preserved")?;
        let contracted = module.function(c"sgd_contracted")?;
        let stream = runtime.create_stream()?;
        assert!(N > 0);
        Ok(Self {
            runtime: runtime.clone(),
            stream,
            preserved,
            contracted,
            weights: [
                runtime.allocate(N * size_of::<f32>())?,
                runtime.allocate(N * size_of::<f32>())?,
                runtime.allocate(N * size_of::<f32>())?,
            ],
            gradient: [
                runtime.allocate(N * size_of::<f32>())?,
                runtime.allocate(N * size_of::<f32>())?,
                runtime.allocate(N * size_of::<f32>())?,
            ],
            count: u32::try_from(N)?,
        })
    }
    pub fn upload(
        &mut self,
        bank: usize,
        weights: &[f32],
        gradient: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        self.weights[bank]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), weights).bytes())?;
        self.gradient[bank]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 1), gradient).bytes())?;
        Ok(())
    }
    pub fn submit(&self, bank: usize, contracted: bool) -> Result<DeviceBuffer, Box<dyn Error>> {
        self.launch(&self.weights[bank], &self.gradient[bank], contracted)
    }
    pub fn host(
        &mut self,
        weights: &[f32],
        gradient: &[f32],
        contracted: bool,
    ) -> Result<DeviceBuffer, Box<dyn Error>> {
        self.upload(2, weights, gradient)?;
        self.submit(2, contracted)
    }
    fn launch(
        &self,
        left: &DeviceBuffer,
        right: &DeviceBuffer,
        contracted: bool,
    ) -> Result<DeviceBuffer, Box<dyn Error>> {
        let output = self
            .runtime
            .allocate(usize::try_from(self.count)? * size_of::<f32>())?;
        let count = self.count.to_ne_bytes();
        let rate = super::oracle::RATE.to_ne_bytes();
        let arguments = [
            HipKernelArgument::Buffer(left),
            HipKernelArgument::Buffer(right),
            HipKernelArgument::Buffer(&output),
            HipKernelArgument::Bytes(&rate),
            HipKernelArgument::Bytes(&count),
        ];
        let kernel = if contracted {
            &self.contracted
        } else {
            &self.preserved
        };
        // SAFETY: ABI is three F32 buffers, F32 rate, u32 count. All owners outlive
        // the guarded launch and terminal event wait; output has one lane per input.
        #[allow(unsafe_code)]
        let mut completion = unsafe {
            kernel.launch(
                &self.stream,
                [self.count.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &arguments,
            )
        }?;
        completion.wait()?;
        Ok(output)
    }
    pub fn read(output: &DeviceBuffer, observed: &mut [f32]) -> Result<(), Box<dyn Error>> {
        output.copy_to(bytemuck::cast_slice_mut(observed))?;
        Ok(())
    }
}

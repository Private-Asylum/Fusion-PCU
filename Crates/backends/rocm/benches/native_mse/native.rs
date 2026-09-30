//! Private squared-difference launch then safe typed ASUM at the production synchronous boundary.
#[rustfmt::skip]
use std::{
    error::Error,
    mem::size_of,
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
    Rocblas,
    compile_hip_source,
};
const SOURCE: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void tensor_native_mse_squared_difference(
    const float *prediction, const float *target, float *squared, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        volatile float difference = prediction[id] - target[id];
        squared[id] = difference * difference;
    }
}
"#;
pub struct Native {
    runtime: HipRuntime,
    stream: HipStreamHandle,
    kernel: HipKernel,
    blas: Rocblas,
    prediction: [DeviceBuffer; 2],
    target: [DeviceBuffer; 2],
    count: u32,
    scale: f32,
}
impl Native {
    pub fn new<const N: usize>(runtime: &HipRuntime) -> Result<Self, Box<dyn Error>> {
        let architecture = runtime
            .device_info()?
            .architecture
            .ok_or("selected device lacks architecture")?;
        let image = compile_hip_source(SOURCE, &architecture)?;
        let kernel = runtime
            .load_module(&image)?
            .function(c"tensor_native_mse_squared_difference")?;
        let stream = runtime.create_stream()?;
        let mut blas = Rocblas::new(runtime)?;
        blas.bind_stream(&stream)?;
        assert!(N > 0 && N <= 1 << 24);
        #[allow(clippy::cast_precision_loss)] // Benchmark extents are exact F32 integers.
        let scale = 1.0 / N as f32;
        Ok(Self {
            runtime: runtime.clone(),
            stream,
            kernel,
            blas,
            prediction: [
                runtime.allocate(N * size_of::<f32>())?,
                runtime.allocate(N * size_of::<f32>())?,
            ],
            target: [
                runtime.allocate(N * size_of::<f32>())?,
                runtime.allocate(N * size_of::<f32>())?,
            ],
            count: u32::try_from(N)?,
            scale,
        })
    }
    pub fn upload(
        &mut self,
        bank: usize,
        prediction: &[f32],
        target: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        self.prediction[bank]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), prediction).bytes())?;
        self.target[bank]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 1), target).bytes())?;
        Ok(())
    }
    pub fn submit(&self, bank: usize) -> Result<DeviceBuffer, Box<dyn Error>> {
        let squared = self
            .runtime
            .allocate(usize::try_from(self.count)? * size_of::<f32>())?;
        let count = self.count.to_ne_bytes();
        let arguments = [
            HipKernelArgument::Buffer(&self.prediction[bank]),
            HipKernelArgument::Buffer(&self.target[bank]),
            HipKernelArgument::Buffer(&squared),
            HipKernelArgument::Bytes(&count),
        ];
        // SAFETY: ABI is three F32 buffers and a u32 count. Every guarded lane fits all buffers;
        // the kernel, stream and owners survive the same terminal event wait as production.
        #[allow(unsafe_code)]
        let mut completion = unsafe {
            self.kernel.launch(
                &self.stream,
                [self.count.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &arguments,
            )
        }?;
        completion.wait()?;
        let output = self.runtime.allocate(size_of::<f32>())?;
        self.blas.sasum_scaled(
            usize::try_from(self.count)?,
            &squared,
            1,
            self.scale,
            &output,
        )?;
        Ok(output)
    }
    pub fn read(output: &DeviceBuffer, observed: &mut [f32]) -> Result<(), Box<dyn Error>> {
        let mut bytes = [0; size_of::<f32>()];
        output.copy_to(&mut bytes)?;
        observed[0] = f32::from_ne_bytes(bytes);
        Ok(())
    }
}

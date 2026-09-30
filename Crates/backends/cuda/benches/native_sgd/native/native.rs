//! Matching native F32 rate/product/subtract or contracted FMA and per-kernel event completion.
#[path = "ffi/ffi.rs"]
mod ffi;
use std::error::Error;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuHostArgument,
    PcuPrecisionPolicy,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    compile_cuda_source_for_device,
    CudaKernel,
    CudaRuntime,
    CudaStreamHandle,
    DeviceBuffer,
};

pub struct Native {
    runtime: CudaRuntime,
    stream: CudaStreamHandle,
    kernel: CudaKernel,
    weights: DeviceBuffer,
    gradient: DeviceBuffer,
    count: u32,
    rate: f32,
}
impl Native {
    pub fn new<const N: usize>(
        runtime: &CudaRuntime,
        precision: PcuPrecisionPolicy,
        rate: f32,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(rate.is_finite() && N > 0);
        let arithmetic = if precision == PcuPrecisionPolicy::BackendOptimized {
            "output[id] = __fmaf_rn(-learning_rate, gradient[id], weights[id]);"
        } else {
            "const float product = __fmul_rn(learning_rate, gradient[id]); output[id] = __fsub_rn(weights[id], product);"
        };
        let source = format!(
            r#"extern "C" __global__ void fusion_kernel(const float* weights, const float* gradient, float* output, float learning_rate, unsigned int count) {{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {{ {arithmetic} }}
}}
"#
        );
        let image = compile_cuda_source_for_device(runtime, &source)?;
        let kernel = runtime.load_module(&image)?.function(c"fusion_kernel")?;
        Ok(Self {
            runtime: runtime.clone(),
            stream: runtime.create_stream()?,
            kernel,
            weights: runtime.allocate(N * size_of::<f32>())?,
            gradient: runtime.allocate(N * size_of::<f32>())?,
            count: u32::try_from(N)?,
            rate,
        })
    }
    pub fn host<const N: usize>(
        &mut self,
        weights: &[f32; N],
        gradient: &[f32; N],
        observed: &mut [f32],
    ) -> Result<(), Box<dyn Error>> {
        assert_eq!(usize::try_from(self.count).unwrap(), N);
        assert_eq!(observed.len(), N);
        self.weights
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), weights).bytes())?;
        self.gradient
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 1), gradient).bytes())?;
        let output = self.runtime.allocate(N * size_of::<f32>())?;
        ffi::update(
            &self.kernel,
            &self.stream,
            self.count,
            self.rate,
            [&self.weights, &self.gradient, &output],
        )?;
        output.copy_to(
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), observed)
                .bytes_mut()
                .unwrap(),
        )?;
        drop(output);
        Ok(())
    }
}

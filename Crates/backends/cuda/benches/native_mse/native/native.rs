//! Two retained inputs, fresh squared scratch and scalar, same stream/configuration and status.
#[path = "ffi/ffi.rs"]
mod ffi;
use std::error::Error;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuHostArgument,
    PcuPrecisionPolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    compile_cuda_source_for_device,
    Cublas,
    CublasEnvironmentSnapshot,
    CublasNumericalConfig,
    CudaKernel,
    CudaRuntime,
    CudaStreamHandle,
    DeviceBuffer,
};

pub struct Native {
    runtime: CudaRuntime,
    blas: Cublas,
    stream: CudaStreamHandle,
    kernel: CudaKernel,
    prediction: DeviceBuffer,
    target: DeviceBuffer,
    count: u32,
}

impl Native {
    pub fn new<const N: usize>(
        runtime: &CudaRuntime,
        precision: PcuPrecisionPolicy,
    ) -> Result<Self, Box<dyn Error>> {
        let count = u32::try_from(N)?;
        let source = format!(
            r#"extern "C" __global__ void fusion_kernel(const float* prediction, const float* target, float* squared) {{
    const unsigned long long id = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= {count}ull) return;
    const float difference = __fsub_rn(prediction[id], target[id]);
    squared[id] = __fmul_rn(difference, difference);
}}
"#
        );
        let image = compile_cuda_source_for_device(runtime, &source)?;
        let module = runtime.load_module(&image)?;
        let kernel = module.function(c"fusion_kernel")?;
        let stream = runtime.create_stream()?;
        let config = CublasNumericalConfig::new(
            PcuScalarType::F32,
            precision,
            CublasEnvironmentSnapshot::capture(),
        )?;
        let mut blas = Cublas::new_with_numerical_config(runtime, config)?;
        blas.bind_stream(&stream)?;
        eprintln!(
            "native MSE admitted config/observed={:?}; no mode retuning or portable reduction claim",
            blas.numerical_config()
        );
        Ok(Self {
            runtime: runtime.clone(),
            blas,
            stream,
            kernel,
            prediction: runtime.allocate(N * size_of::<f32>())?,
            target: runtime.allocate(N * size_of::<f32>())?,
            count,
        })
    }

    pub fn host<const N: usize>(
        &mut self,
        prediction: &[f32; N],
        target: &[f32; N],
        observed: &mut [f32; 1],
    ) -> Result<(), Box<dyn Error>> {
        assert_eq!(usize::try_from(self.count).unwrap(), N);
        self.prediction
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), prediction).bytes())?;
        self.target
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 1), target).bytes())?;
        let squared = self.runtime.allocate(N * size_of::<f32>())?;
        ffi::squared(
            &self.kernel,
            &self.stream,
            self.count,
            &self.prediction,
            &self.target,
            &squared,
        )?;
        let output = self.runtime.allocate(size_of::<f32>())?;
        self.blas
            .sasum_scaled(N, &squared, 1, super::oracle::scale(N), &output)?;
        drop(squared);
        output.copy_to(
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), observed)
                .bytes_mut()
                .unwrap(),
        )?;
        drop(output);
        Ok(())
    }
}

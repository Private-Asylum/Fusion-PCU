//! Independently launched identical native F64 loss with matching fresh output/event boundary.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
use std::{
    error::Error,
    mem::size_of,
    time::{
        Duration,
        Instant,
    },
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuHostArgument,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    ValueId,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    HipRuntime,
    HipKernel,
    HipStreamHandle,
    DeviceBuffer,
    compile_hip_source_for_device,
    lower_native_mse_to_hip_source,
};
use super::oracle::Scalar;
pub struct Native {
    runtime: HipRuntime,
    kernel: HipKernel,
    stream: HipStreamHandle,
    weights: [DeviceBuffer; 2],
    gradient: [DeviceBuffer; 2],
    output_bytes: usize,
    count: u32,
    scale: f64,
    blas: Rocblas,
}
impl Native {
    pub fn new<T: Scalar, const N: usize>(
        runtime: &HipRuntime,
        graph: &Graph,
        output: ValueId,
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(T::TYPE, PcuScalarType::F64);
        let node = graph.node(output)?;
        let source = lower_native_mse_to_hip_source(graph, output)?;
        let image = compile_hip_source_for_device(runtime, &source)?;
        let module = runtime.load_module(&image)?;
        let stream = runtime.create_stream()?;
        // Typed DASUM/DSCAL offer is F64 for either permitted precision setting.
        let mut blas = Rocblas::new(runtime)?;
        blas.bind_stream(&stream)?;
        let _ = node;
        Ok(Self {
            runtime: runtime.clone(),
            kernel: module.function(c"tensor_native_mse_squared_difference")?,
            stream,
            weights: [
                runtime.allocate(N * size_of::<T>())?,
                runtime.allocate(N * size_of::<T>())?,
            ],
            gradient: [
                runtime.allocate(N * size_of::<T>())?,
                runtime.allocate(N * size_of::<T>())?,
            ],
            output_bytes: size_of::<T>(),
            count: u32::try_from(N)?,
            scale: 1.0 / f64::from(u32::try_from(N)?),
            blas,
        })
    }
    pub fn upload<T: Scalar>(
        &mut self,
        bank: usize,
        weights: &[T],
        gradient: &[T],
    ) -> Result<(), Box<dyn Error>> {
        self.weights[bank]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), weights).bytes())?;
        self.gradient[bank]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 1), gradient).bytes())?;
        Ok(())
    }
    fn launch(
        &self,
        bank: usize,
        output: &DeviceBuffer,
    ) -> Result<fusion_pcu_rocm::HipCompletion, Box<dyn Error>> {
        ffi::launch(
            &self.kernel,
            &self.stream,
            self.count,
            [&self.weights[bank], &self.gradient[bank], output],
        )
        .map_err(Into::into)
    }
    pub fn submit(&self, bank: usize) -> Result<DeviceBuffer, Box<dyn Error>> {
        let squared = self
            .runtime
            .allocate(usize::try_from(self.count)? * size_of::<f64>())?;
        self.launch(bank, &squared)?.wait()?;
        let output = self.runtime.allocate(self.output_bytes)?;
        self.blas.dasum_scaled(
            usize::try_from(self.count)?,
            &squared,
            1,
            self.scale,
            &output,
        )?;
        drop(squared);
        Ok(output)
    }
    pub fn phases(&self) -> Result<(Duration, Duration), Box<dyn Error>> {
        let start = Instant::now();
        let squared = self
            .runtime
            .allocate(usize::try_from(self.count)? * size_of::<f64>())?;
        let mut completion = self.launch(0, &squared)?;
        let submission = start.elapsed();
        let start = Instant::now();
        completion.wait()?;
        let output = self.runtime.allocate(self.output_bytes)?;
        self.blas.dasum_scaled(
            usize::try_from(self.count)?,
            &squared,
            1,
            self.scale,
            &output,
        )?;
        drop(squared);
        drop(output);
        Ok((submission, start.elapsed()))
    }
    pub fn read<T: Scalar>(
        output: &DeviceBuffer,
        observed: &mut [T],
    ) -> Result<(), Box<dyn Error>> {
        output.copy_to(
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), observed)
                .bytes_mut()
                .unwrap(),
        )?;
        Ok(())
    }
}

#[rustfmt::skip]
use fusion_pcu_rocm::{
    Rocblas,
};

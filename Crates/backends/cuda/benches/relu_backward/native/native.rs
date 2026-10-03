//! Independent native launch of the identical checked selection and completion work.
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
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    ValueId,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaRuntime,
    CudaKernel,
    CudaStreamHandle,
    DeviceBuffer,
    compile_cuda_source_for_device,
    lower_relu_backward_to_cuda_source,
};
use super::oracle::Scalar;
pub struct Native {
    runtime: CudaRuntime,
    kernel: CudaKernel,
    stream: CudaStreamHandle,
    weights: [DeviceBuffer; 2],
    gradient: [DeviceBuffer; 2],
    output_bytes: usize,
    grid: u32,
}
impl Native {
    pub fn new<T: Scalar, const N: usize>(
        runtime: &CudaRuntime,
        graph: &Graph,
        output: ValueId,
    ) -> Result<Self, Box<dyn Error>> {
        let source = lower_relu_backward_to_cuda_source(graph, output)?;
        let image = compile_cuda_source_for_device(runtime, &source)?;
        let module = runtime.load_module(&image)?;
        Ok(Self {
            runtime: runtime.clone(),
            kernel: module.function(c"fusion_kernel")?,
            stream: runtime.create_stream()?,
            weights: [
                runtime.allocate(N * size_of::<T>())?,
                runtime.allocate(N * size_of::<T>())?,
            ],
            gradient: [
                runtime.allocate(N * size_of::<T>())?,
                runtime.allocate(N * size_of::<T>())?,
            ],
            output_bytes: N * size_of::<T>(),
            grid: u32::try_from(N)?.div_ceil(256),
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
    fn destinations(&self) -> Result<(DeviceBuffer, DeviceBuffer), Box<dyn Error>> {
        let output = self.runtime.allocate(self.output_bytes)?;
        let mut status = self.runtime.allocate(size_of::<u64>())?;
        status.copy_from(&u64::MAX.to_le_bytes())?;
        Ok((output, status))
    }
    fn launch(
        &self,
        bank: usize,
        output: &DeviceBuffer,
        status: &DeviceBuffer,
    ) -> Result<fusion_pcu_cuda::CudaCompletion, Box<dyn Error>> {
        ffi::launch(
            &self.kernel,
            &self.stream,
            self.grid,
            self.output_bytes,
            [&self.weights[bank], &self.gradient[bank], output, status],
        )
        .map_err(Into::into)
    }
    pub fn submit(&self, bank: usize) -> Result<(DeviceBuffer, u64), Box<dyn Error>> {
        let (output, status) = self.destinations()?;
        self.launch(bank, &output, &status)?.wait()?;
        let mut bytes = [0; 8];
        status.copy_to(&mut bytes)?;
        Ok((output, u64::from_le_bytes(bytes)))
    }
    pub fn phases(&self) -> Result<(Duration, Duration), Box<dyn Error>> {
        let start = Instant::now();
        let (output, status) = self.destinations()?;
        let mut completion = self.launch(0, &output, &status)?;
        let submission = start.elapsed();
        let start = Instant::now();
        completion.wait()?;
        let mut bytes = [0; 8];
        status.copy_to(&mut bytes)?;
        assert_eq!(u64::from_le_bytes(bytes), u64::MAX);
        drop(status);
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

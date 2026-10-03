//! The identical fixed checked tensor kernel, independently launched with a fresh output.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
use std::error::Error;
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
    lower_checked_float_tensor_to_cuda_source,
};
use super::oracle::Format;
pub struct Native {
    runtime: CudaRuntime,
    kernel: CudaKernel,
    stream: CudaStreamHandle,
    inputs: [[DeviceBuffer; 2]; 2],
    bytes: usize,
    grid: [u32; 3],
    unary: bool,
}
impl Native {
    pub fn new<T: Format, const N: usize>(
        runtime: &CudaRuntime,
        graph: &Graph,
        value: ValueId,
    ) -> Result<Self, Box<dyn Error>> {
        let source = lower_checked_float_tensor_to_cuda_source(graph, value)?;
        let image = compile_cuda_source_for_device(runtime, &source)?;
        let module = runtime.load_module(&image)?;
        Ok(Self {
            runtime: runtime.clone(),
            unary: matches!(
                graph.node(value)?.op,
                fusion_pcu::dialect::tensor::OpDescriptor::Relu { .. }
            ),
            kernel: module.function(c"fusion_kernel")?,
            stream: runtime.create_stream()?,
            inputs: [
                [
                    runtime.allocate(N * size_of::<T>())?,
                    runtime.allocate(N * size_of::<T>())?,
                ],
                [
                    runtime.allocate(N * size_of::<T>())?,
                    runtime.allocate(N * size_of::<T>())?,
                ],
            ],
            bytes: N * size_of::<T>(),
            grid: [u32::try_from(N)?.div_ceil(256), 1, 1],
        })
    }
    pub fn upload<T: Format>(
        &mut self,
        bank: usize,
        a: &[T],
        b: &[T],
    ) -> Result<(), Box<dyn Error>> {
        for (buffer, data) in self.inputs[bank].iter_mut().zip([a, b]) {
            buffer.copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), data).bytes())?;
        }
        Ok(())
    }
    pub fn submit(&self, bank: usize) -> Result<DeviceBuffer, Box<dyn Error>> {
        let output = self.runtime.allocate(self.bytes)?;
        // The owned executor allocates a private checked status for each fresh result.
        // Match that physical boundary rather than retaining an advantageous native status.
        let mut status = self.runtime.allocate(8)?;
        status.copy_from(&u64::MAX.to_le_bytes())?;
        let unary_buffers = [&self.inputs[bank][0], &output, &status];
        let binary_buffers = [
            &self.inputs[bank][0],
            &self.inputs[bank][1],
            &output,
            &status,
        ];
        let buffers = if self.unary {
            &unary_buffers[..]
        } else {
            &binary_buffers[..]
        };
        ffi::launch(&self.kernel, &self.stream, self.grid, [256, 1, 1], buffers)?.wait()?;
        let mut word = [0; 8];
        status.copy_to(&mut word)?;
        assert_eq!(u64::from_le_bytes(word), u64::MAX);
        Ok(output)
    }
    pub fn read<T: Format>(
        output: &DeviceBuffer,
        observed: &mut [T],
    ) -> Result<(), Box<dyn Error>> {
        let bytes = output.len();
        output.copy_to(
            &mut PcuHostArgument::read_write(PcuBindingRef::new(0, 2), observed)
                .bytes_mut()
                .unwrap()[..bytes],
        )?;
        Ok(())
    }
}

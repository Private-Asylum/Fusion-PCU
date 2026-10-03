//! Direct native launch of the same admitted kernel with matched retained storage.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
use std::num::NonZeroU32;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuDispatchSubmission,
    PcuHostArgument,
    PcuInvocationShape,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmOwnedDispatchBackend,
    HipKernel,
    HipStreamHandle,
    DeviceBuffer,
};
use super::oracle::Format;
pub struct Native {
    kernel: HipKernel,
    stream: HipStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    inputs: Vec<[DeviceBuffer; 1]>,
    output: [DeviceBuffer; 1],
    prefix_bytes: usize,
}
impl Native {
    pub fn new<T: Format, const N: usize>(
        backend: &RocmOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        bank_count: usize,
    ) -> Self {
        assert_eq!(ir.bindings.len(), 2);
        let prepared = backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: ir,
                shape: PcuInvocationShape::invocations(
                    NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
                ),
            })
            .unwrap();
        let (grid, block) = prepared.launch_geometry();
        let mut output =
            std::array::from_fn(|_| backend.allocate((N + 2) * size_of::<T>()).unwrap());
        let initial = vec![T::sentinel(); N + 2];
        for buffer in &mut output {
            buffer
                .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), &initial).bytes())
                .unwrap();
        }
        Self {
            kernel: prepared.hip_kernel(),
            stream: prepared.stream_handle(),
            grid,
            block,
            inputs: (0..bank_count)
                .map(|_| std::array::from_fn(|_| backend.allocate(N * size_of::<T>()).unwrap()))
                .collect(),
            output,
            prefix_bytes: N * size_of::<T>(),
        }
    }
    pub fn upload<T: Format>(&mut self, bank: usize, input: &[T]) {
        for (buffer, values) in self.inputs[bank].iter_mut().zip([input]) {
            buffer
                .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), values).bytes())
                .unwrap();
        }
    }
    pub fn submit(&self, bank: usize) {
        ffi::launch(
            &self.kernel,
            &self.stream,
            self.grid,
            self.block,
            [&self.inputs[bank][0], &self.output[0]],
        )
        .unwrap()
        .wait()
        .unwrap();
    }
    pub fn read<T: Format>(&self, output: &mut [T]) {
        for (buffer, values) in self.output.iter().zip([output]) {
            let mut argument = PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values);
            buffer
                .copy_to(&mut argument.bytes_mut().unwrap()[..self.prefix_bytes])
                .unwrap();
        }
    }
    pub fn read_resident<T: Format>(&self, output: &mut [T]) {
        for (buffer, values) in self.output.iter().zip([output]) {
            buffer
                .copy_to(
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values)
                        .bytes_mut()
                        .unwrap(),
                )
                .unwrap();
        }
    }
    pub fn host<T: Format>(&mut self, input: &[T], output: &mut [T]) {
        self.upload(0, input);
        self.submit(0);
        self.read(output);
    }
}

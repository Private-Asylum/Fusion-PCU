//! Direct native launch of the same admitted kernel with matched retained status and storage.
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
use fusion_pcu_cuda::{
    CudaOwnedDispatchBackend,
    CudaKernel,
    CudaStreamHandle,
    DeviceBuffer,
};
use super::oracle::Integer;
pub struct Native {
    kernel: CudaKernel,
    stream: CudaStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    inputs: Vec<[DeviceBuffer; 2]>,
    output: [DeviceBuffer; 2],
    status: DeviceBuffer,
    sentinel: bool,
    prefix_bytes: usize,
}
impl Native {
    pub fn new<T: Integer, const N: usize>(
        backend: &CudaOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        bank_count: usize,
    ) -> Self {
        assert_eq!(ir.bindings.len(), 4);
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
        let initial = vec![T::SENTINEL; N + 2];
        for buffer in &mut output {
            buffer
                .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), &initial).bytes())
                .unwrap();
        }
        Self {
            kernel: prepared.cuda_kernel(),
            stream: prepared.stream_handle(),
            grid,
            block,
            inputs: (0..bank_count)
                .map(|_| std::array::from_fn(|_| backend.allocate(N * size_of::<T>()).unwrap()))
                .collect(),
            output,
            status: backend.allocate(size_of::<u64>()).unwrap(),
            sentinel: false,
            prefix_bytes: N * size_of::<T>(),
        }
    }
    pub fn upload<T: Integer>(&mut self, bank: usize, lhs: &[T], rhs: &[T]) {
        for (buffer, values) in self.inputs[bank].iter_mut().zip([lhs, rhs]) {
            buffer
                .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), values).bytes())
                .unwrap();
        }
    }
    pub fn submit(&mut self, bank: usize) -> u64 {
        if !self.sentinel {
            self.status.copy_from(&u64::MAX.to_le_bytes()).unwrap();
        }
        ffi::launch(
            &self.kernel,
            &self.stream,
            self.grid,
            self.block,
            [
                &self.inputs[bank][0],
                &self.inputs[bank][1],
                &self.output[0],
                &self.output[1],
                &self.status,
            ],
        )
        .unwrap()
        .wait()
        .unwrap();
        let mut word = [0; 8];
        self.status.copy_to(&mut word).unwrap();
        let word = u64::from_le_bytes(word);
        self.sentinel = word == u64::MAX;
        word
    }
    pub fn read<T: Integer>(&self, quotient: &mut [T], remainder: &mut [T]) {
        for (buffer, values) in self.output.iter().zip([quotient, remainder]) {
            let mut argument = PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values);
            buffer
                .copy_to(&mut argument.bytes_mut().unwrap()[..self.prefix_bytes])
                .unwrap();
        }
    }
    pub fn read_resident<T: Integer>(&self, quotient: &mut [T], remainder: &mut [T]) {
        for (buffer, values) in self.output.iter().zip([quotient, remainder]) {
            buffer
                .copy_to(
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values)
                        .bytes_mut()
                        .unwrap(),
                )
                .unwrap();
        }
    }
    pub fn host<T: Integer>(&mut self, lhs: &[T], rhs: &[T], q: &mut [T], r: &mut [T]) {
        self.upload(0, lhs, rhs);
        assert_eq!(self.submit(0), u64::MAX);
        self.read(q, r);
    }
}

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
    PcuPreparedOwnedDispatch,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaOwnedDispatchBackend,
    CudaKernel,
    CudaStreamHandle,
    DeviceBuffer,
};
use super::oracle::Format;
pub struct Native {
    kernel: CudaKernel,
    stream: CudaStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    inputs: Vec<[DeviceBuffer; 1]>,
    output: [DeviceBuffer; 1],
    status: DeviceBuffer,
    sentinel: bool,
    prefix_bytes: usize,
}
impl Native {
    pub fn new<T: Format, const N: usize>(
        backend: &CudaOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        bank_count: usize,
    ) -> Self {
        assert!((2..=3).contains(&ir.bindings.len()));
        let prepared = backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: ir,
                shape: PcuInvocationShape::invocations(
                    NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
                ),
            })
            .unwrap();
        assert_eq!(
            prepared.binding_schema().len(),
            2,
            "unused declarations must not occupy the device ABI"
        );
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
            kernel: prepared.cuda_kernel(),
            stream: prepared.stream_handle(),
            grid,
            block,
            inputs: (0..bank_count)
                .map(|_| {
                    std::array::from_fn(|_| {
                        backend
                            .allocate(
                                if ir.bindings.len() == 3 && ir.bindings[2].name == Some("seed") {
                                    size_of::<T>()
                                } else {
                                    N * size_of::<T>()
                                },
                            )
                            .unwrap()
                    })
                })
                .collect(),
            output,
            status: backend.allocate(size_of::<u64>()).unwrap(),
            sentinel: false,
            prefix_bytes: N * size_of::<T>(),
        }
    }
    pub fn upload<T: Format>(&mut self, bank: usize, lhs: &[T]) {
        for (buffer, values) in self.inputs[bank].iter_mut().zip([lhs]) {
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
            [&self.output[0], &self.inputs[bank][0], &self.status],
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
    pub fn read<T: Format>(&self, output: &mut [T]) {
        for (buffer, values) in self.output.iter().zip([output]) {
            let mut argument = PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values);
            buffer
                .copy_to(&mut argument.bytes_mut().unwrap()[..self.prefix_bytes])
                .unwrap();
        }
    }
    pub fn host<T: Format>(&mut self, lhs: &[T], output: &mut [T]) {
        self.upload(0, lhs);
        assert_eq!(self.submit(0), u64::MAX);
        self.read(output);
    }
}

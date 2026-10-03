//! Minimal retained three-pointer native transport, with the same two host outputs.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
use fusion_pcu::{
 PcuDispatchKernelIr, PcuDispatchSubmission, PcuInvocationShape,
 PcuBindingRef, PcuHostArgument, PcuPreparedOwnedDispatch,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{CudaOwnedDispatchBackend,CudaKernel,CudaStreamHandle,DeviceBuffer};
use super::oracle::Format;
pub struct Native {
    kernel: CudaKernel,
    stream: CudaStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    input: DeviceBuffer,
    outputs: [DeviceBuffer; 2],
}
impl Native {
    pub fn new<T: Format, const N: usize>(
        backend: &CudaOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
    ) -> Self {
        let prepared = backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: ir,
                shape: PcuInvocationShape::invocations(
                    std::num::NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
                ),
            })
            .unwrap();
        fusion_pcu::describe_scalar_transport_map::<4>(ir, T::TYPE).unwrap();
        assert_eq!(
            prepared
                .binding_schema()
                .iter()
                .map(|b| b.target.binding)
                .collect::<Vec<_>>(),
            [0, 2, 3]
        );
        let (grid, block) = prepared.launch_geometry();
        Self {
            kernel: prepared.cuda_kernel(),
            stream: prepared.stream_handle(),
            grid,
            block,
            input: backend.allocate(N * size_of::<T>()).unwrap(),
            outputs: std::array::from_fn(|_| backend.allocate(N * size_of::<T>()).unwrap()),
        }
    }
    pub fn host<T: Format>(&mut self, input: &[T], stage: &mut [T], output: &mut [T]) {
        let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
        self.input.copy_from(bytes.bytes()).unwrap();
        ffi::launch(
            &self.kernel,
            &self.stream,
            self.grid,
            self.block,
            [&self.input, &self.outputs[0], &self.outputs[1]],
        )
        .unwrap()
        .wait()
        .unwrap();
        let readbacks = self
            .outputs
            .each_ref()
            .map(|buffer| buffer.readback_owned_at(0, bytes.bytes().len()).unwrap());
        for (ticket, values) in readbacks.iter().zip([stage, output]) {
            ticket.publish_to(
                &mut PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values)
                    .bytes_mut()
                    .unwrap()[..bytes.bytes().len()],
            );
        }
    }
}

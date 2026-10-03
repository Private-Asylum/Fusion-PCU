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
    output: [DeviceBuffer; 2],
    status: DeviceBuffer,
    sentinel: bool,
    roles: [u8; 3],
    prefix_bytes: usize,
}
impl Native {
    pub fn new<T: Format, const N: usize>(
        backend: &CudaOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        bank_count: usize,
    ) -> Self {
        assert_eq!(ir.bindings.len(), 3);
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
            3,
            "both ordered outputs and the actual input must occupy the ABI"
        );
        let roles = std::array::from_fn(|slot| {
            u8::try_from(prepared.binding_schema()[slot].target.binding).unwrap()
        });
        assert_eq!(
            roles
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            [0, 1, 2].into_iter().collect()
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
                .map(|_| std::array::from_fn(|_| backend.allocate(N * size_of::<T>()).unwrap()))
                .collect(),
            output,
            status: backend.allocate(size_of::<u64>()).unwrap(),
            sentinel: false,
            roles,
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
        ffi::launch(&self.kernel, &self.stream, self.grid, self.block, {
            let buffers = self.roles.map(|role| match role {
                0 => &self.output[0],
                1 => &self.output[1],
                2 => &self.inputs[bank][0],
                _ => unreachable!(),
            });
            [buffers[0], buffers[1], buffers[2], &self.status]
        })
        .unwrap()
        .wait()
        .unwrap();
        let mut word = [0; 8];
        self.status.copy_to(&mut word).unwrap();
        let word = u64::from_le_bytes(word);
        self.sentinel = word == u64::MAX;
        word
    }
    pub fn read<T: Format>(&self, stage: &mut [T], output: &mut [T]) {
        self.read_spans(stage, output, [self.prefix_bytes; 2]);
    }
    fn read_spans<T: Format>(&self, stage: &mut [T], output: &mut [T], spans: [usize; 2]) {
        let readbacks = std::array::from_fn::<_, 2, _>(|index| {
            self.output[index]
                .readback_owned_at(0, spans[index])
                .unwrap()
        });
        for ((ticket, values), span) in readbacks.iter().zip([stage, output]).zip(spans) {
            ticket.publish_to(
                &mut PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values)
                    .bytes_mut()
                    .unwrap()[..span],
            );
        }
    }

    pub fn host<T: Format>(&mut self, input: &[T], stage: &mut [T], output: &mut [T]) {
        // Stage is read/write in the captured ABI even though its first operation overwrites it.
        // Preserve the prepared/source H2D boundary rather than optimizing this control alone.
        self.output[0]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), stage).bytes())
            .unwrap();
        self.upload(0, input);
        assert_eq!(self.submit(0), u64::MAX);
        // Historical baseline diagnostic keeps its full stage view. Both readbacks still
        // finish privately before either caller output is published.
        self.read_spans(
            stage,
            output,
            [std::mem::size_of_val(stage), self.prefix_bytes],
        );
    }
    /// Minimal native work: stage is always overwritten before its same-lane read.
    pub fn minimal_host<T: Format>(&mut self, input: &[T], stage: &mut [T], output: &mut [T]) {
        self.upload(0, input);
        assert_eq!(self.submit(0), u64::MAX);
        self.read(stage, output);
    }
    pub fn read_full<T: Format>(&self, stage: &mut [T], output: &mut [T]) {
        self.read_spans(
            stage,
            output,
            [std::mem::size_of_val(stage), std::mem::size_of_val(output)],
        );
    }
}

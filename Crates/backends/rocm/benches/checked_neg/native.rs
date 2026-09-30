//! Manual launch of the admitted unary checker, retaining terminal-success status.
use std::num::NonZeroU32;

#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuDispatchSubmission,
    PcuInvocationShape,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    HipKernel,
    HipKernelArgument,
    RocmOwnedDispatchBackend,
    HipStreamHandle,
    DeviceBuffer,
};

pub struct Native {
    kernel: HipKernel,
    stream: HipStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    input: DeviceBuffer,
    output: DeviceBuffer,
    status: DeviceBuffer,
    sentinel: bool,
    readback: Vec<u8>,
}

impl Native {
    pub fn new(
        backend: &RocmOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        bytes: usize,
    ) -> Self {
        let prepared = backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: ir,
                shape: PcuInvocationShape::invocations(
                    NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
                ),
            })
            .unwrap();
        let (grid, block) = prepared.launch_geometry();
        eprintln!("native checked Neg geometry: grid={grid:?}, block={block:?}");
        Self {
            kernel: prepared.hip_kernel(),
            stream: prepared.stream_handle(),
            grid,
            block,
            input: backend.allocate(bytes).unwrap(),
            output: backend.allocate(bytes).unwrap(),
            status: backend.allocate(size_of::<u64>()).unwrap(),
            sentinel: false,
            readback: vec![0; bytes],
        }
    }

    pub fn upload(&mut self, bytes: &[u8]) {
        self.input.copy_from(bytes).unwrap();
    }

    pub fn submit(&mut self) -> u64 {
        if !self.sentinel {
            self.status.copy_from(&u64::MAX.to_ne_bytes()).unwrap();
        }
        let arguments = [
            HipKernelArgument::Buffer(&self.input),
            HipKernelArgument::Buffer(&self.output),
            HipKernelArgument::Buffer(&self.status),
        ];
        // SAFETY: the admitted unary map has exactly these input/output/status buffers; both
        // typed payload allocations cover every logical lane. The selected prepared executable
        // supplies matching geometry and all resources survive terminal event completion.
        #[allow(unsafe_code)]
        // Exact admitted kernel ABI and retained buffers are documented above.
        unsafe {
            self.kernel
                .launch(&self.stream, self.grid, self.block, 0, &arguments)
        }
        .unwrap()
        .wait()
        .unwrap();
        let mut status = [0u8; size_of::<u64>()];
        self.status.copy_to(&mut status).unwrap();
        let word = u64::from_ne_bytes(status);
        self.sentinel = word == u64::MAX;
        word
    }

    pub fn download(&mut self) -> &[u8] {
        self.output.copy_to(&mut self.readback).unwrap();
        &self.readback
    }

    pub fn host(&mut self, bytes: &[u8]) {
        self.upload(bytes);
        assert_eq!(self.submit(), u64::MAX);
        std::hint::black_box(self.download());
    }

    pub fn completed_bytes(&self) -> &[u8] {
        &self.readback
    }
}

//! Manual CUDA launch of the exact PCU-lowered source, with the checked status contract.
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchSubmission,
    PcuInvocationShape,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaKernelArgument,
    CudaOwnedDispatchBackend,
    CudaKernel,
    CudaStreamHandle,
    DeviceBuffer,
};
use std::num::NonZeroU32;

pub struct Native {
    kernel: CudaKernel,
    stream: CudaStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    inputs: [DeviceBuffer; 2],
    output: DeviceBuffer,
    status: DeviceBuffer,
    sentinel: bool,
    readback: Vec<u8>,
}

impl Native {
    pub fn new<const N: usize>(backend: &CudaOwnedDispatchBackend) -> Self {
        let bindings = super::source::transform_bindings();
        let builder = super::source::transform_ir::<N>(&bindings).unwrap();
        let prepared = builder
            .with_ir(|kernel| {
                backend.prepare_dispatch(PcuDispatchSubmission {
                    kernel,
                    shape: PcuInvocationShape::invocations(
                        NonZeroU32::new(u32::try_from(N).unwrap()).unwrap(),
                    ),
                })
            })
            .unwrap();
        let (grid, block) = prepared.launch_geometry();
        eprintln!("native matched geometry: N={N}, grid={grid:?}, block={block:?}");
        Self {
            kernel: prepared.cuda_kernel(),
            stream: prepared.stream_handle(),
            grid,
            block,
            inputs: std::array::from_fn(|_| backend.allocate(N * size_of::<f32>()).unwrap()),
            output: backend.allocate(N * size_of::<f32>()).unwrap(),
            status: backend.allocate(size_of::<u64>()).unwrap(),
            sentinel: false,
            readback: vec![0; N * size_of::<f32>()],
        }
    }

    pub fn upload(&mut self, bytes: &[u8]) {
        self.upload_bank(0, bytes);
    }

    pub fn upload_bank(&mut self, bank: usize, bytes: &[u8]) {
        self.inputs[bank].copy_from(bytes).unwrap();
    }

    pub fn submit(&mut self) -> u64 {
        self.submit_bank(0)
    }

    pub fn submit_bank(&mut self, bank: usize) -> u64 {
        if !self.sentinel {
            self.status.copy_from(&u64::MAX.to_le_bytes()).unwrap();
        }
        let arguments = [
            CudaKernelArgument::Buffer(&self.inputs[bank]),
            CudaKernelArgument::Buffer(&self.output),
            CudaKernelArgument::Buffer(&self.status),
        ];
        // SAFETY: transform_ir defines precisely two F32 buffers followed by checked U64
        // status; allocations cover its exact N, and geometry comes from that same prepared
        // executable. This mirrors the backend's checked_dispatch benchmark launch pattern.
        unsafe {
            self.kernel
                .launch(&self.stream, self.grid, self.block, 0, &arguments)
        }
        .unwrap()
        .wait()
        .unwrap();
        let mut word = [0; 8];
        self.status.copy_to(&mut word).unwrap();
        let value = u64::from_le_bytes(word);
        self.sentinel = value == u64::MAX;
        value
    }

    pub fn readback(&mut self) -> &[u8] {
        self.output.copy_to(&mut self.readback).unwrap();
        &self.readback
    }

    pub fn result(&self) -> &[u8] {
        &self.readback
    }

    pub fn host(&mut self, bytes: &[u8]) {
        self.upload(bytes);
        assert_eq!(self.submit(), u64::MAX);
        std::hint::black_box(self.readback());
    }
}

//! Direct native execution of the source identity kernel, with explicit staging boundaries.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchSubmission,
    PcuInvocationShape,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaKernel,
    CudaOwnedDispatchBackend,
    transfer::CudaPinnedBuffer,
    CudaRuntime,
    CudaStreamHandle,
    DeviceBuffer,
};
use std::num::NonZeroU32;
use super::source;

pub struct Native {
    kernel: CudaKernel,
    stream: CudaStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    input: DeviceBuffer,
    output: DeviceBuffer,
    upload: [Option<CudaPinnedBuffer>; 2],
    download: Option<CudaPinnedBuffer>,
}

impl Native {
    pub fn new<const N: usize>(backend: &CudaOwnedDispatchBackend, runtime: &CudaRuntime) -> Self {
        let declarations = source::copy_bindings();
        let builder = source::copy_ir::<N>(&declarations).unwrap();
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
        let upload = [17_u8, 239].map(|value| {
            let mut buffer = runtime.allocate_pinned(N).unwrap();
            buffer.as_bytes_mut().fill(value);
            Some(buffer)
        });
        Self {
            kernel: prepared.cuda_kernel(),
            stream: prepared.stream_handle(),
            grid,
            block,
            input: backend.allocate(N).unwrap(),
            output: backend.allocate(N).unwrap(),
            upload,
            download: Some(runtime.allocate_pinned(N).unwrap()),
        }
    }

    fn launch(&self) {
        ffi::launch(
            &self.kernel,
            &self.stream,
            self.grid,
            self.block,
            [&self.input, &self.output],
        )
        .unwrap()
        .wait()
        .unwrap();
    }

    pub fn pageable(&mut self, input: &[u8], output: &mut [u8]) {
        self.input.copy_from(input).unwrap();
        self.launch();
        self.output.copy_to(output).unwrap();
    }

    pub fn pinned(&mut self, bank: usize) {
        self.upload[bank] = Some(
            self.upload[bank]
                .take()
                .unwrap()
                .upload_reusable(&self.input, &self.stream)
                .unwrap()
                .finish()
                .unwrap(),
        );
        self.launch();
        self.download = Some(
            self.download
                .take()
                .unwrap()
                .download(&self.output, &self.stream)
                .unwrap()
                .finish()
                .unwrap(),
        );
    }

    pub fn pinned_result(&self) -> &[u8] {
        self.download.as_ref().unwrap().as_bytes()
    }
}

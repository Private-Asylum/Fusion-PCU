//! Private native comparison ABI and terminal owner-retaining launch boundary.
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaError,
    CudaKernel,
    CudaKernelArgument,
    CudaStreamHandle,
    CudaCompletion,
    DeviceBuffer,
};
pub(super) fn launch(
    kernel: &CudaKernel,
    stream: &CudaStreamHandle,
    grid: u32,
    bytes: usize,
    buffers: [&DeviceBuffer; 4],
) -> Result<CudaCompletion, CudaError> {
    assert!(
        bytes > 0
            && buffers[..2].iter().all(|buffer| !buffer.is_empty())
            && buffers[2].len() >= bytes
            && buffers[3].len() == 8
    );
    let arguments = buffers.map(CudaKernelArgument::Buffer);
    // SAFETY: Native builds only the validated three typed storage-pointer/u64-status ABI.
    // Extents cover all lanes; launch retains buffer/module/stream owners until completion.
    unsafe { kernel.launch(stream, [grid, 1, 1], [256, 1, 1], 0, &arguments) }
}

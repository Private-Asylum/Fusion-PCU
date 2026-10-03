//! Exact generated storage-pointer ABI, retained through terminal completion.
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaCompletion,
    CudaError,
    CudaKernel,
    CudaKernelArgument,
    CudaStreamHandle,
    DeviceBuffer,
};
pub(super) fn launch(
    kernel: &CudaKernel,
    stream: &CudaStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    buffers: [&DeviceBuffer; 4],
) -> Result<CudaCompletion, CudaError> {
    let arguments = buffers.map(CudaKernelArgument::Buffer);
    // SAFETY: cold source preparation fixes this pointer ABI and exact allocation extents.
    // The caller retains matching buffers; returned completion retains launch/module leases.
    unsafe { kernel.launch(stream, grid, block, 0, &arguments) }
}

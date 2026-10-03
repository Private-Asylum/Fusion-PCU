//! Exact generated three-storage-pointer plus U64 checked-status ABI.
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
    grid: [u32; 3],
    block: [u32; 3],
    buffers: [&DeviceBuffer; 4],
) -> Result<CudaCompletion, CudaError> {
    let args = buffers.map(CudaKernelArgument::Buffer);
    // SAFETY: The admitted source IR declares three exact-width storage pointers, then the
    // private U64 fault pointer. Cold preparation froze ABI/geometry; safe allocations and
    // completion retain every buffer/module/stream. No foreign ABI lives outside this seam.
    unsafe { kernel.launch(stream, grid, block, 0, &args) }
}

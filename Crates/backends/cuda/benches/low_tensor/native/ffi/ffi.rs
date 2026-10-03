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
    buffers: &[&DeviceBuffer],
) -> Result<CudaCompletion, CudaError> {
    let mut args = [
        CudaKernelArgument::Buffer(buffers[0]),
        CudaKernelArgument::Buffer(buffers[1]),
        CudaKernelArgument::Buffer(buffers[2]),
        CudaKernelArgument::Buffer(buffers[2]),
    ];
    if let Some(buffer) = buffers.get(3) {
        args[3] = CudaKernelArgument::Buffer(buffer);
    }
    // SAFETY: The admitted source IR declares two or three exact-width storage pointers, then the
    // private U64 fault pointer. Cold preparation froze ABI/geometry; safe allocations and
    // completion retain every buffer/module/stream. No foreign ABI lives outside this seam.
    unsafe { kernel.launch(stream, grid, block, 0, &args[..buffers.len()]) }
}

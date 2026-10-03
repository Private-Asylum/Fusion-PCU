//! Exact generated three-storage-pointer plus U64 checked-status ABI.
#[rustfmt::skip]
use fusion_pcu_rocm::{
    HipError,
    HipKernel,
    HipKernelArgument,
    HipStreamHandle,
    HipCompletion,
    DeviceBuffer,
};
pub(super) fn launch(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    buffers: &[&DeviceBuffer],
) -> Result<HipCompletion, HipError> {
    let mut args = [
        HipKernelArgument::Buffer(buffers[0]),
        HipKernelArgument::Buffer(buffers[1]),
        HipKernelArgument::Buffer(buffers[2]),
        HipKernelArgument::Buffer(buffers[2]),
    ];
    if let Some(buffer) = buffers.get(3) {
        args[3] = HipKernelArgument::Buffer(buffer);
    }
    // SAFETY: The admitted source IR declares two or three exact-width storage pointers, then the
    // private U64 fault pointer. Cold preparation froze ABI/geometry; safe allocations and
    // completion retain every buffer/module/stream. No foreign ABI lives outside this seam.
    unsafe { kernel.launch(stream, grid, block, 0, &args[..buffers.len()]) }
}

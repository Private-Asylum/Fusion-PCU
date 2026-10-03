//! Exact generated two-storage-pointer identity ABI.
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
    buffers: [&DeviceBuffer; 2],
) -> Result<HipCompletion, HipError> {
    let args = buffers.map(HipKernelArgument::Buffer);
    // SAFETY: The admitted source IR declares two exact-width storage pointers. Cold preparation froze ABI/geometry; safe allocations and
    // completion retain every buffer/module/stream. No foreign ABI lives outside this seam.
    unsafe { kernel.launch(stream, grid, block, 0, &args) }
}

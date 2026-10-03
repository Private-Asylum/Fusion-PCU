//! Exact generated storage-pointer ABI, retained through terminal completion.
#[rustfmt::skip]
use fusion_pcu_rocm::{
    HipCompletion,
    HipError,
    HipKernel,
    HipKernelArgument,
    HipStreamHandle,
    DeviceBuffer,
};
pub(super) fn launch(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    buffers: [&DeviceBuffer; 4],
) -> Result<HipCompletion, HipError> {
    let arguments = buffers.map(HipKernelArgument::Buffer);
    // SAFETY: cold source preparation fixes this pointer ABI and exact allocation extents.
    // The caller retains matching buffers; returned completion retains launch/module leases.
    unsafe { kernel.launch(stream, grid, block, 0, &arguments) }
}

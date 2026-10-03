//! Private native comparison ABI and terminal owner-retaining launch boundary.
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
    grid: u32,
    bytes: usize,
    buffers: [&DeviceBuffer; 4],
) -> Result<HipCompletion, HipError> {
    assert!(
        bytes > 0
            && buffers[..3].iter().all(|buffer| buffer.len() >= bytes)
            && buffers[3].len() == 8
    );
    let arguments = buffers.map(HipKernelArgument::Buffer);
    // SAFETY: Native builds only the validated three typed storage-pointer/u64-status ABI.
    // Extents cover all lanes; launch retains buffer/module/stream owners until completion.
    unsafe { kernel.launch(stream, [grid, 1, 1], [256, 1, 1], 0, &arguments) }
}

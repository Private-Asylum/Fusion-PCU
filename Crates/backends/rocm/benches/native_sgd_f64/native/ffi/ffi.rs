//! Private typed native F64 kernel ABI; foreign launch ownership stays in this seam.
#[rustfmt::skip]
use fusion_pcu_rocm::{HipError, HipKernel, HipKernelArgument, HipStreamHandle, HipCompletion, DeviceBuffer};
pub(super) fn launch(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    count: u32,
    rate: f64,
    buffers: [&DeviceBuffer; 3],
) -> Result<HipCompletion, HipError> {
    let bytes = usize::try_from(count).unwrap() * size_of::<f64>();
    assert!(count > 0 && rate.is_finite() && buffers.iter().all(|buffer| buffer.len() >= bytes));
    let rate_bytes = rate.to_ne_bytes();
    let count_bytes = count.to_ne_bytes();
    let args = [
        HipKernelArgument::Buffer(buffers[0]),
        HipKernelArgument::Buffer(buffers[1]),
        HipKernelArgument::Buffer(buffers[2]),
        HipKernelArgument::Bytes(&rate_bytes),
        HipKernelArgument::Bytes(&count_bytes),
    ];
    // SAFETY: Validated admitted code has three F64 storage pointers, exact widened rate,
    // U32 count and no numerical status. Typed completion retains every launch owner.
    unsafe { kernel.launch(stream, [count.div_ceil(256), 1, 1], [256, 1, 1], 0, &args) }
}

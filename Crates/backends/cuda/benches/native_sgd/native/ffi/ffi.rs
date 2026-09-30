//! Private unsafe kernel ABI boundary; SDK calls remain behind the typed backend wrappers.
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaError,
    CudaKernel,
    CudaKernelArgument,
    CudaStreamHandle,
    DeviceBuffer,
};

pub(super) fn update(
    kernel: &CudaKernel,
    stream: &CudaStreamHandle,
    count: u32,
    rate: f32,
    buffers: [&DeviceBuffer; 3],
) -> Result<(), CudaError> {
    let bytes = usize::try_from(count).unwrap() * size_of::<f32>();
    assert!(count > 0 && rate.is_finite() && buffers.iter().all(|buffer| buffer.len() >= bytes));
    let rate_bytes = rate.to_ne_bytes();
    let count_bytes = count.to_ne_bytes();
    let arguments = [
        CudaKernelArgument::Buffer(buffers[0]),
        CudaKernelArgument::Buffer(buffers[1]),
        CudaKernelArgument::Buffer(buffers[2]),
        CudaKernelArgument::Bytes(&rate_bytes),
        CudaKernelArgument::Bytes(&count_bytes),
    ];
    // SAFETY: Native creates this exact three F32-pointer, frozen F32-rate, U32-count ABI.
    // Checked extents cover all lanes; the typed launch retains module/stream/buffer owners
    // through its per-kernel terminal event. No raw SDK symbol or invented batch is used.
    unsafe {
        kernel.launch(
            stream,
            [count.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &arguments,
        )
    }?
    .wait()
}

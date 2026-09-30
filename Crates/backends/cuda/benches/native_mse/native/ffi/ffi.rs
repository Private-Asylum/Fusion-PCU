//! Private unsafe kernel ABI boundary; all SDK calls remain in the backend's FFI implementation.
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaError,
    CudaKernel,
    CudaKernelArgument,
    CudaStreamHandle,
    DeviceBuffer,
};

pub(super) fn squared(
    kernel: &CudaKernel,
    stream: &CudaStreamHandle,
    count: u32,
    prediction: &DeviceBuffer,
    target: &DeviceBuffer,
    output: &DeviceBuffer,
) -> Result<(), CudaError> {
    let bytes = usize::try_from(count).unwrap() * size_of::<f32>();
    assert!(
        count > 0
            && [prediction, target, output]
                .iter()
                .all(|buffer| buffer.len() >= bytes)
    );
    let arguments = [
        CudaKernelArgument::Buffer(prediction),
        CudaKernelArgument::Buffer(target),
        CudaKernelArgument::Buffer(output),
    ];
    // SAFETY: private Native constructs this kernel from the exact three-pointer F32 ABI below.
    // Bounds cover every lane; resources and module survive terminal completion. The safe
    // backend launch wrapper also checks physical runtime/stream and retains allocation leases.
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

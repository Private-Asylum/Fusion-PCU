//! The sole unsafe native launch endpoint; retained runtime owners guard completion.
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipError,
    HipKernel,
    HipKernelArgument,
    HipStreamHandle,
};

pub fn launch(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    input: &DeviceBuffer,
    output: &DeviceBuffer,
    bytes: u32,
) -> Result<(), HipError> {
    let extent = bytes.to_ne_bytes();
    let arguments = [
        HipKernelArgument::Buffer(input),
        HipKernelArgument::Buffer(output),
        HipKernelArgument::Bytes(&extent),
    ];
    // SAFETY: handwritten byte_identity has exactly two allocation pointers then a U32
    // extent. Both allocations cover that extent; completion retains their owners.
    unsafe {
        kernel.launch(
            stream,
            [bytes.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &arguments,
        )
    }?
    .wait()
}

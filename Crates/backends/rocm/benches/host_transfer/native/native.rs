//! Independent handwritten HIP identity using public HIP runtime resource wrappers.
#[path = "ffi/ffi.rs"]
mod ffi;
use std::error::Error;
#[rustfmt::skip]
use fusion_pcu_rocm::{
    compile_hip_source_for_device,
    DeviceBuffer,
    HipError,
    HipKernel,
    HipRuntime,
    HipStreamHandle,
};

const SOURCE: &str = r#"
extern "C" __global__ void byte_identity(
    const unsigned char *input, unsigned char *output, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = input[id];
}
"#;

pub struct Native {
    input: DeviceBuffer,
    output: DeviceBuffer,
    kernel: HipKernel,
    stream: HipStreamHandle,
    bytes: u32,
}

impl Native {
    pub fn prepare(runtime: &HipRuntime, bytes: usize) -> Result<Self, Box<dyn Error>> {
        let image = compile_hip_source_for_device(runtime, SOURCE)?;
        let module = runtime.load_module(&image)?;
        Ok(Self {
            input: runtime.allocate(bytes)?,
            output: runtime.allocate(bytes)?,
            kernel: module.function(c"byte_identity")?,
            stream: runtime.create_stream()?,
            bytes: u32::try_from(bytes).expect("benchmark extent fits the HIP ABI"),
        })
    }

    pub fn execute(&mut self, input: &[u8], output: &mut [u8]) -> Result<(), HipError> {
        assert_eq!(
            input.len(),
            usize::try_from(self.bytes).expect("HIP extent fits the host address space")
        );
        assert!(output.len() >= input.len());
        self.input.copy_from(input)?;
        ffi::launch(
            &self.kernel,
            &self.stream,
            &self.input,
            &self.output,
            self.bytes,
        )?;
        self.output.copy_to(&mut output[..input.len()])
    }
}

//! Resident native HIP peer for the Add -> `ReLU` fusion benchmark.

use std::error::Error;

use fusion_pcu::PcuObjectRef;
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    RocmDiscovery,
    compile_hip_source_for_device,
};

const SOURCE: &str = r#"
extern "C" __global__ void native_add_relu(
    const float *left, const float *right, float *output, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float sum = left[id] + right[id];
        output[id] = fmaxf(sum, 0.0f);
    }
}
"#;

/// Native one-kernel Add -> `ReLU` with the same resident input/output boundary as PCU.
pub struct NativeAddRelu {
    _runtime: HipRuntime,
    stream: HipStreamHandle,
    kernel: HipKernel,
    left: DeviceBuffer,
    right: DeviceBuffer,
    output: DeviceBuffer,
    elements: u32,
}

impl NativeAddRelu {
    pub fn prepare(
        discovery: &RocmDiscovery,
        device: PcuObjectRef,
        left: &[f32],
        right: &[f32],
    ) -> Result<Self, Box<dyn Error>> {
        if left.is_empty() || left.len() != right.len() {
            return Err("native Add+ReLU input shapes must be equal and nonempty".into());
        }
        let elements = u32::try_from(left.len())?;
        let runtime = discovery.open_device(device)?;
        let image = compile_hip_source_for_device(&runtime, SOURCE)?;
        let module = runtime.load_module(&image)?;
        let kernel = module.function(c"native_add_relu")?;
        let stream = runtime.create_stream()?;
        let bytes = std::mem::size_of_val(left);
        let mut left_buffer = runtime.allocate(bytes)?;
        let mut right_buffer = runtime.allocate(bytes)?;
        let output = runtime.allocate(bytes)?;
        left_buffer.copy_from(bytemuck::cast_slice(left))?;
        right_buffer.copy_from(bytemuck::cast_slice(right))?;
        Ok(Self {
            _runtime: runtime,
            stream,
            kernel,
            left: left_buffer,
            right: right_buffer,
            output,
            elements,
        })
    }

    pub fn execute_resident(&self) -> Result<(), Box<dyn Error>> {
        let count_bytes = self.elements.to_ne_bytes();
        let arguments = [
            HipKernelArgument::Buffer(&self.left),
            HipKernelArgument::Buffer(&self.right),
            HipKernelArgument::Buffer(&self.output),
            HipKernelArgument::Bytes(&count_bytes),
        ];
        // SAFETY: Both inputs and output hold `elements` f32 values. The bounds guard excludes
        // padded lanes, and completion is awaited before any buffer can be released.
        #[allow(unsafe_code)]
        let mut completion = unsafe {
            self.kernel.launch(
                &self.stream,
                [self.elements.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &arguments,
            )?
        };
        completion.wait()?;
        Ok(())
    }

    pub fn readback(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut values = vec![0.0; self.elements as usize];
        self.output.copy_to(bytemuck::cast_slice_mut(&mut values))?;
        Ok(values)
    }
}

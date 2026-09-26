//! Check whether the active HIPRTC defaults contract an SGD multiply-subtract.
//!
//! This is a hardware/compiler witness, not a conformance guarantee for every `ROCm` release. It
//! compares the unqualified expression to a forced separate-rounding path and explicit `fmaf`.

use std::error::Error;

use fusion_pcu_rocm::{
    HipKernelArgument,
    HipRuntime,
    compile_hip_source_for_device,
};

const SOURCE: &str = r#"
extern "C" __global__ void fma_witness(const float* weights, const float* rate,
                                       const float* gradient, float* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // The expression whose default HIPRTC contraction behavior we want to observe.
        output[0] = weights[0] - rate[0] * gradient[0];

        // Volatile forces the product to be rounded before the subtraction.
        volatile float product = rate[0] * gradient[0];
        output[1] = weights[0] - product;

        // The same arithmetic with an explicit fused operation.
        output[2] = __builtin_fmaf(-rate[0], gradient[0], weights[0]);
    }
}
"#;

fn main() -> Result<(), Box<dyn Error>> {
    let devices = HipRuntime::enumerate_devices()?;
    let device = devices.first().ok_or("HIP reports no visible device")?;
    let runtime = HipRuntime::new(u32::try_from(device.index)?)?;
    let code = compile_hip_source_for_device(&runtime, SOURCE)?;
    let module = runtime.load_module(&code)?;
    let kernel = module.function(c"fma_witness")?;

    let weights = 1.0_f32;
    let rate = f32::from_bits(1.0_f32.to_bits() + 1);
    let gradient = f32::from_bits(1.0_f32.to_bits() - 2);
    let mut weight_buffer = runtime.allocate(4)?;
    let mut rate_buffer = runtime.allocate(4)?;
    let mut gradient_buffer = runtime.allocate(4)?;
    let output_buffer = runtime.allocate(12)?;
    weight_buffer.copy_from(&weights.to_ne_bytes())?;
    rate_buffer.copy_from(&rate.to_ne_bytes())?;
    gradient_buffer.copy_from(&gradient.to_ne_bytes())?;
    let stream = runtime.create_stream()?;
    let arguments = [
        HipKernelArgument::Buffer(&weight_buffer),
        HipKernelArgument::Buffer(&rate_buffer),
        HipKernelArgument::Buffer(&gradient_buffer),
        HipKernelArgument::Buffer(&output_buffer),
    ];
    // SAFETY: The kernel ABI is four f32 pointers, and each allocation has sufficient storage.
    // The launch has one thread, which writes each output exactly once; completion is awaited.
    #[allow(unsafe_code)]
    let mut completion = unsafe { kernel.launch(&stream, [1, 1, 1], [1, 1, 1], 0, &arguments)? };
    completion.wait()?;

    let mut bytes = [0_u8; 12];
    output_buffer.copy_to(&mut bytes)?;
    let results = bytes
        .chunks_exact(4)
        .map(|word| f32::from_ne_bytes(word.try_into().expect("four-byte result")))
        .collect::<Vec<_>>();
    let [default_expression, strict_separate, explicit_fma] = results.as_slice() else {
        return Err("kernel returned an unexpected output count".into());
    };

    println!(
        "device: {} ({})",
        device.name,
        device
            .architecture
            .as_deref()
            .unwrap_or("unknown architecture")
    );
    println!("HIPRTC options: none (runtime defaults)");
    println!("weights={weights:?} (0x{:08x})", weights.to_bits());
    println!("rate={rate:?} (0x{:08x})", rate.to_bits());
    println!("gradient={gradient:?} (0x{:08x})", gradient.to_bits());
    println!(
        "default weights - rate * gradient: {default_expression:?} (0x{:08x})",
        default_expression.to_bits()
    );
    println!(
        "forced separate multiply/subtract: {strict_separate:?} (0x{:08x})",
        strict_separate.to_bits()
    );
    println!(
        "explicit fmaf(-rate, gradient, weights): {explicit_fma:?} (0x{:08x})",
        explicit_fma.to_bits()
    );
    println!(
        "default expression contracted: {}",
        default_expression.to_bits() == explicit_fma.to_bits()
            && default_expression.to_bits() != strict_separate.to_bits()
    );
    Ok(())
}

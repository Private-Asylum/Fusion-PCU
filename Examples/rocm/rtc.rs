//! Compile a Rust-authored PCU kernel through HIPRTC for a runtime-selected HIP device.

#[path = "selection.rs"]
mod selection;

use std::error::Error;

use fusion_pcu_macros::pcu;
use fusion_pcu_rocm::{
    HipKernelArgument,
    RocmDiscovery,
    compile_hip_source_for_device,
    lower_dispatch_to_hip_rtc_source,
};

#[pcu(invocations = 250)]
fn rtc_probe<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id] + 1.0;
        id += stride;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let bindings = rtc_probe_bindings();
    let builder = rtc_probe::<2048>(&bindings)?;
    let kernel = builder.ir();
    let source = lower_dispatch_to_hip_rtc_source(&kernel)?;
    let discovery = RocmDiscovery::new();
    let candidates = selection::ranked_devices(&discovery, selection::preferred_device()?, false)?;
    let mut failures = Vec::new();
    for candidate in candidates {
        let result = (|| -> Result<(), Box<dyn Error>> {
            let runtime = discovery.open_device(candidate.device)?;
            let image = compile_hip_source_for_device(&runtime, &source)?;
            let module = runtime.load_module(&image)?;
            let kernel = module.function(c"fusion_kernel")?;
            let input_values = (0..2048_u16).map(f32::from).collect::<Vec<_>>();
            let input_bytes = input_values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            let mut input = runtime.allocate(input_bytes.len())?;
            let output = runtime.allocate(input_bytes.len())?;
            input.copy_from(&input_bytes)?;
            let stream = runtime.create_stream()?;
            let arguments = [
                HipKernelArgument::Buffer(&input),
                HipKernelArgument::Buffer(&output),
            ];
            // SAFETY: The lowered kernel ABI is exactly two f32 pointers in binding order.
            // Both allocations contain 2048 f32 elements. Four 64-thread blocks cover the 250
            // logical invocations; the lowered guard rejects padded lanes. Completion is awaited.
            #[allow(unsafe_code)]
            let mut completion =
                unsafe { kernel.launch(&stream, [4, 1, 1], [64, 1, 1], 0, &arguments)? };
            completion.wait()?;
            let mut output_bytes = vec![0_u8; input_bytes.len()];
            output.copy_to(&mut output_bytes)?;
            for (index, word) in output_bytes.chunks_exact(4).enumerate() {
                let actual = f32::from_ne_bytes(word.try_into()?);
                let expected = input_values[index] + 1.0;
                if actual.to_bits() != expected.to_bits() {
                    return Err(
                        format!("HIPRTC output[{index}]={actual}; expected {expected}").into(),
                    );
                }
            }
            println!(
                "HIPRTC compiled, loaded, and verified PCU Dispatch for device {} ({})",
                candidate.device.id, candidate.name
            );
            Ok(())
        })();
        match result {
            Ok(()) => return Ok(()),
            Err(error) => failures.push(format!("device {}: {error}", candidate.device.id)),
        }
    }
    Err(format!(
        "no ROCm device compiled this kernel through HIPRTC: {}",
        failures.join("; ")
    )
    .into())
}

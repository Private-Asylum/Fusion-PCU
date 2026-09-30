use fusion_pcu::pcu;

#[pcu(invocations = N)]
fn transform<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 2.0 + 1.0;
}

fn main() -> Result<(), fusion_pcu::PcuExecutionError> {
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Cuda,
        ..fusion_pcu::global::PcuExecutionPolicy::default()
    })?;
    let input = [1.0_f32, -2.0, 3.5];
    let mut output = [0.0_f32; 3];
    transform::<3>(&input, &mut output)?;
    println!("CUDA source transform: {output:?}");
    Ok(())
}

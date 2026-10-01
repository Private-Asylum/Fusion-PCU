//! Headless prepared checked arithmetic with ordinary per-function source.

use fusion_pcu_vulkan::PcuVulkanBackend;
use pcu_facade::pcu;

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate_f64<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = PcuVulkanBackend::new()?;
    let mut call = negate_prepare::<5, _>(&backend)?;
    let input = [1.0, -2.0, 0.0, -0.0, f32::from_bits(1)];
    let mut output = [17.0; 7];
    call(&input, &mut output)?;
    for (value, actual) in input.iter().zip(&output) {
        assert_eq!(actual.to_bits(), value.to_bits() ^ 0x8000_0000);
    }
    assert_eq!(&output[5..], &[17.0; 2]);
    println!(
        "{}: checked Neg {:?}, untouched tail {:?}",
        backend.name(),
        &output[..5],
        &output[5..]
    );
    if backend.caps().shader_float64 {
        let mut call = negate_f64_prepare::<5, _>(&backend)?;
        let input = [1.0_f64, -2.0, 0.0, -0.0, f64::from_bits(1)];
        let mut output = [17.0_f64; 7];
        call(&input, &mut output)?;
        for (value, actual) in input.iter().zip(&output) {
            assert_eq!(actual.to_bits(), value.to_bits() ^ 0x8000_0000_0000_0000);
        }
        assert_eq!(&output[5..], &[17.0; 2]);
        println!(
            "{}: F64 checked Neg {:?}, untouched tail {:?}",
            backend.name(),
            &output[..5],
            &output[5..]
        );
    }
    Ok(())
}

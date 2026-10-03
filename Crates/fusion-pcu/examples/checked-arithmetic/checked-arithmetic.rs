//! Ordinary Rust call boundaries, checked arithmetic and lossless wide transport.
//! Run: cargo run -p fusion-pcu --features cpu --example checked-arithmetic
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuCheckedFloat,
    PcuCheckedInteger,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuScalar,
    PcuU512,
};

#[pcu(invocations = N)]
fn transform<T: PcuCheckedFloat, const N: usize>(
    input: &[T; N],
    scale: &[T; N],
    output: &mut [T; N],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * scale[id];
}

#[pcu(invocations = N)]
fn preserve<T: PcuScalar, const N: usize>(input: &[T; N], output: &mut [T; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[pcu(invocations = N)]
fn sum<T: PcuCheckedInteger, const N: usize>(left: &[T; N], right: &[T; N], output: &mut [T; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}

fn main() -> Result<(), PcuExecutionError> {
    // CPU must be enabled and explicitly selected. No unavailable GPU silently falls back.
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;
    // Ordinary stack borrows describe read-only input and exclusive output access.
    // PCU chooses staging for the selected backend; CPU reads RAM directly here.
    // Success publishes output before returning. Checked faults preserve caller output.
    let input = [1.25, -2.0, f64::from_bits(1), -0.0];
    let mut output = [0.0; 4];
    transform(&input, &[2.0; 4], &mut output)?;
    println!("RAM output: {output:?}");
    let previous = output.map(f64::to_bits);
    let fault = transform(&[f64::MAX; 4], &[2.0; 4], &mut output).unwrap_err();
    assert_eq!(output.map(f64::to_bits), previous);
    println!("Checked overflow: {fault}");
    // Recognizing and transporting a 512-bit value does not imply GPU arithmetic.
    let wide = PcuU512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, 8]);
    let mut copied = [PcuU512::ZERO; 2];
    preserve(&[wide; 2], &mut copied)?;
    assert_eq!(copied, [wide; 2]);
    println!("Lossless wide RAM output: {copied:?}");
    // The CPU also admits checked wide arithmetic. High limbs are actual values,
    // not metadata, and overflow has the same Result/rollback contract as u32.
    let mut doubled = [PcuU512::ZERO; 2];
    sum(&copied, &copied, &mut doubled)?;
    assert_eq!(
        doubled,
        [PcuU512::from_limbs_le([2, 4, 6, 8, 10, 12, 14, 16]); 2]
    );
    let previous = doubled;
    let fault = sum(&[PcuU512::MAX; 2], &[wide; 2], &mut doubled).unwrap_err();
    assert_eq!(doubled, previous);
    println!("Checked 512-bit overflow: {fault}");
    // Reference narrowing checks the actual binary64 input at half precision;
    // it does not first round through binary32. These constants are exactly representable.
    // The same annotated function now executes the independently qualified CPU half profile.
    // Its integer-based implementation does not depend on native half instructions.
    let half = PcuF16Bits::pcu_checked_from_f64(1.25)
        .expect("the constant 1.25 is exactly representable in binary16");
    let mut half_output = [PcuF16Bits::from_bits(0); 2];
    transform(
        &[half; 2],
        &[PcuF16Bits::from_bits(0x4000); 2],
        &mut half_output,
    )?;
    assert_eq!(half_output.map(PcuF16Bits::to_bits), [0x4100; 2]);
    println!(
        "Checked half RAM output: {:?}",
        half_output.map(PcuF16Bits::to_bits)
    );
    // OFP8 uses its named encoding and PCU's checked reference rounding/fault rules.
    // The identical generic source call executes checked FP8 arithmetic on CPU.
    // Other providers must admit their own exact format/operation profile independently.
    let quantized = PcuF8E4M3FnBits::pcu_checked_from_f64(1.25)
        .expect("the constant 1.25 is exactly representable in OFP8 E4M3FN");
    let mut fp8_output = [PcuF8E4M3FnBits::from_bits(0); 2];
    transform(
        &[quantized; 2],
        &[PcuF8E4M3FnBits::from_bits(0x40); 2],
        &mut fp8_output,
    )?;
    assert_eq!(fp8_output.map(PcuF8E4M3FnBits::to_bits), [0x42; 2]);
    println!(
        "Checked FP8 RAM output: {:?}",
        fp8_output.map(PcuF8E4M3FnBits::to_bits)
    );
    // No escaped device owner is returned. Per-call access has quiesced;
    // prepared implementation state may remain cached for subsequent calls.
    Ok(())
}

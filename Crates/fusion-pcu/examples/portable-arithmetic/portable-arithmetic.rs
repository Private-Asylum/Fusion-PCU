//! Bounded portable arithmetic using ordinary Rust borrows and stack output.
//! Run: cargo run -p fusion-pcu --features cpu --example portable-arithmetic
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};

// PortableV1 currently closes exactly one Add/Sub/Mul/Div map over F16/BF16/OFP8.
// It is independent of Strict. These native/precision permissions demonstrate
// that a scalar's explicit checked operation still cannot lose its checks.
// Unsupported formats or compound graphs return an error before submission.
#[pcu(invocations = N, flag(deterministic), flag(strict), flag(native_compound), flag(backend_precision))]
fn divide<T: PcuCheckedFloat, const N: usize>(left: &[T; N], right: &[T; N], output: &mut [T; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}

fn exercise<T: PcuCheckedFloat + PartialEq + core::fmt::Debug>(
    one: T,
    two: T,
    zero: T,
    half: T,
) -> Result<(), PcuExecutionError> {
    let input = [one; 7];
    let mut divisors = [two; 7];
    let mut output = [zero; 7];
    // CPU reads these stack borrows directly. A selected device provider stages
    // according to its allocation contract and completes before publishing RAM.
    // No upload, event or device lifetime call is required at this boundary.
    divide(&input, &divisors, &mut output)?;
    assert_eq!(output, [half; 7]);
    println!(
        "Portable RAM output for {}: {output:?}",
        core::any::type_name::<T>()
    );
    divisors[4] = zero;
    let fault = divide(&input, &divisors, &mut output).unwrap_err();
    let PcuExecutionError::ArithmeticFault(arithmetic) = fault else {
        return Err(fault);
    };
    assert_eq!(arithmetic.kind, PcuExecutionFaultKind::DivideByZero);
    assert_eq!(arithmetic.invocation_id, 4);
    assert_eq!(output, [half; 7]); // The complete previous output survives.
    divisors[4] = two;
    divide(&input, &divisors, &mut output)?; // A failed call does not poison reuse.
    Ok(())
}

fn main() -> Result<(), PcuExecutionError> {
    // CPU is default-off and is deliberately selected here; no GPU fallback.
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;
    exercise(
        PcuF16Bits::from_bits(0x3c00),
        PcuF16Bits::from_bits(0x4000),
        PcuF16Bits::from_bits(0),
        PcuF16Bits::from_bits(0x3800),
    )?;
    exercise(
        PcuBf16Bits::from_bits(0x3f80),
        PcuBf16Bits::from_bits(0x4000),
        PcuBf16Bits::from_bits(0),
        PcuBf16Bits::from_bits(0x3f00),
    )?;
    exercise(
        PcuF8E4M3FnBits::from_bits(0x38),
        PcuF8E4M3FnBits::from_bits(0x40),
        PcuF8E4M3FnBits::from_bits(0),
        PcuF8E4M3FnBits::from_bits(0x30),
    )?;
    exercise(
        PcuF8E5M2Bits::from_bits(0x3c),
        PcuF8E5M2Bits::from_bits(0x40),
        PcuF8E5M2Bits::from_bits(0),
        PcuF8E5M2Bits::from_bits(0x38),
    )?;
    Ok(())
}

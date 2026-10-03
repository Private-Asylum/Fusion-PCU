//! Explicit CPU selection and observable recovered Clamp output from genuine generic source.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuF16Bits,
    PcuF8E4M3FnBits,
};
#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(clamp_range))]
fn quotient<T: PcuCheckedFloat>(left: &[T; 3], right: &[T; 3], output: &mut [T; 3]) {
    let index = context.global_invocation_id;
    output[index] = left[index] / right[index];
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let mut output = [0.0_f32; 3];
    let error = quotient(&[f32::MAX, 1.0, -f32::MAX], &[0.5; 3], &mut output).unwrap_err();
    assert!(
        matches!(error, global::PcuExecutionError::ArithmeticFault(fault) if fault.recovered && fault.invocation_id == 0)
    );
    assert_eq!(
        output.map(f32::to_bits),
        [f32::MAX, 2.0, -f32::MAX].map(f32::to_bits)
    );
    let before = output;
    assert!(
        matches!(quotient(&[f32::MAX, 1.0, f32::MAX], &[0.5, 0.0, 0.5], &mut output), Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id == 1)
    );
    assert_eq!(output.map(f32::to_bits), before.map(f32::to_bits));
    let mut half = [PcuF16Bits::from_bits(0); 3];
    assert!(
        matches!(quotient(&[PcuF16Bits::from_bits(0x7bff); 3], &[PcuF16Bits::from_bits(0x3800); 3], &mut half), Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert_eq!(half, [PcuF16Bits::from_bits(0x7bff); 3]);
    let mut fp8 = [PcuF8E4M3FnBits::from_bits(0); 3];
    assert!(
        matches!(quotient(&[PcuF8E4M3FnBits::from_bits(0x7e); 3], &[PcuF8E4M3FnBits::from_bits(0x30); 3], &mut fp8), Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert_eq!(fp8, [PcuF8E4M3FnBits::from_bits(0x7e); 3]);
    println!(
        "Recovered Clamp publishes complete useful output; fatal division preserves it. F32/F16/FP8 verified."
    );
}

//! Exact low-format unary source publication and fatal rollback.
#[path = "../../tests/low_unary/source/source.rs"]
#[allow(dead_code)] // Full operation/policy family is tested elsewhere.
mod source;
#[rustfmt::skip]
use pcu_facade::{global,PcuF16Bits,PcuF8E4M3FnBits};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;
    let input = [
        PcuF16Bits::from_bits(1),
        PcuF16Bits::from_bits(0x8000),
        PcuF16Bits::from_bits(0x3c00),
    ];
    let mut output = [PcuF16Bits::from_bits(17); 5];
    assert!(
        matches!(source::neg_clamp_strict::<PcuF16Bits,3>(&input,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert_eq!(output.map(PcuF16Bits::to_bits), [0x8001, 0, 0xbc00, 17, 17]);
    let before = output;
    let mut bad = input;
    bad[2] = PcuF16Bits::from_bits(0x7e01);
    assert!(
        matches!(source::neg_clamp_strict::<PcuF16Bits,3>(&bad,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered&&fault.invocation_id==2)
    );
    assert_eq!(output, before);
    let input = [
        PcuF8E4M3FnBits::from_bits(0x81),
        PcuF8E4M3FnBits::from_bits(1),
        PcuF8E4M3FnBits::from_bits(0x38),
    ];
    let mut output = [PcuF8E4M3FnBits::from_bits(17); 5];
    assert!(
        matches!(source::relu_clamp_strict::<PcuF8E4M3FnBits,3>(&input,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered&&fault.invocation_id==1)
    );
    assert_eq!(output.map(PcuF8E4M3FnBits::to_bits), [0, 1, 0x38, 17, 17]);
    println!(
        "CPU half Neg and named FP8 ReLU retain exact payloads, recovered Err, fatal rollback and tails"
    );
    global::clear_thread_cache()?;
    global::use_defaults()?;
    Ok(())
}

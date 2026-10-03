//! Observable FP8 recovery and fatal rollback through genuine annotated source.
#[path = "../../tests/clamped_binary/source/source.rs"]
#[allow(dead_code)]
mod source;
use pcu_facade::{global, PcuF8E4M3FnBits};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })?;
    let max = PcuF8E4M3FnBits::from_bits(0x7e);
    let one = PcuF8E4M3FnBits::from_bits(0x38);
    let half = PcuF8E4M3FnBits::from_bits(0x30);
    let mut output = [one; 5];
    let left = [max, one, max];
    let mut right = [half; 3];
    assert!(
        matches!(source::div::<PcuF8E4M3FnBits,3>(&left,&right,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert_eq!(
        output.map(PcuF8E4M3FnBits::to_bits),
        [0x7e, 0x40, 0x7e, 0x38, 0x38]
    );
    let before = output;
    right[2] = PcuF8E4M3FnBits::from_bits(0);
    assert!(
        matches!(source::div::<PcuF8E4M3FnBits,3>(&left,&right,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered&&fault.invocation_id==2)
    );
    assert_eq!(output, before);
    println!("Packed Vulkan FP8 recovered output and later fatal rollback: {output:?}");
    global::clear_thread_cache()?;
    global::use_defaults()?;
    Ok(())
}

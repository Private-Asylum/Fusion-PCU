//! Exact F32/F64 unary source with observable recovered errors and fatal rollback.
#[path = "../../tests/portable_unary/source/source.rs"]
#[allow(dead_code)] // Full policy family is separately qualified.
mod source;
use pcu_facade::global;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;
    let input = [f64::from_bits(1), -0.0, 1.0];
    let mut output = [17.0; 5];
    assert!(
        matches!(source::neg_clamp_strict::<f64,3>(&input,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert_eq!(
        output.map(f64::to_bits),
        [f64::from_bits(0x8000_0000_0000_0001), 0.0, -1.0, 17.0, 17.0].map(f64::to_bits)
    );
    let before = output;
    let mut bad = input;
    bad[2] = f64::NAN;
    assert!(
        matches!(source::neg_clamp_strict::<f64,3>(&bad,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered&&fault.invocation_id==2)
    );
    assert_eq!(output.map(f64::to_bits), before.map(f64::to_bits));
    let input = [-0.0_f32, -1.0, f32::from_bits(1)];
    let mut output = [17.0; 5];
    assert!(
        matches!(source::relu_clamp_strict::<f32,3>(&input,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered&&fault.invocation_id==2)
    );
    assert_eq!(
        output.map(f32::to_bits),
        [0.0, 0.0, f32::from_bits(1), 17.0, 17.0].map(f32::to_bits)
    );
    println!("CPU Portable unary recovered payloads and fatal rollback verified");
    global::clear_thread_cache()?;
    global::use_defaults()?;
    Ok(())
}

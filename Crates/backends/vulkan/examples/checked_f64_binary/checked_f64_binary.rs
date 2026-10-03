//! Recovered range errors carry useful output; fatal errors preserve caller memory.
#[path = "../../tests/clamped_binary/source/source.rs"]
#[allow(dead_code)] // Shared annotated operation family; example exercises division.
mod source;
use pcu_facade::global;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })?;
    let left = [f64::MAX, 1.0, f64::MAX];
    let right = [0.5; 3];
    let mut output = [17.0; 5];
    assert!(
        matches!(source::div::<f64,3>(&left,&right,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert_eq!(
        output.map(f64::to_bits),
        [f64::MAX, 2.0, f64::MAX, 17.0, 17.0].map(f64::to_bits)
    );
    let before = output;
    let mut bad = right;
    bad[2] = 0.0;
    assert!(
        matches!(source::div::<f64,3>(&left,&bad,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered&&fault.invocation_id==2)
    );
    assert_eq!(output.map(f64::to_bits), before.map(f64::to_bits));
    println!(
        "Vulkan Clamp publishes useful recovered output and preserves memory on later fatal faults: {output:?}"
    );
    global::clear_thread_cache()?;
    global::use_defaults()?;
    Ok(())
}

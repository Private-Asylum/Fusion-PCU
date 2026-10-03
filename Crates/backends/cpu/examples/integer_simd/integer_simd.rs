//! Genuine checked source retains whole-call publication with vectorized native-width Add/Sub.
#[path = "../../benches/integer_simd/source/source.rs"]
#[allow(dead_code)]
mod source;
use pcu_facade::global;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;
    let mut left = [10_i32; 17];
    let right = [2_i32; 17];
    let mut output = [77_i32; 20];
    source::add::<i32, 17>(&left, &right, &mut output)?;
    assert_eq!(output[..17], [12; 17]);
    left[7] = i32::MAX;
    let saved = output;
    let failure = source::add::<i32, 17>(&left, &right, &mut output).unwrap_err();
    let fault = failure.arithmetic_fault().unwrap();
    assert!(!fault.recovered && fault.invocation_id == 7);
    assert_eq!(output, saved);
    let recovered = source::add_clamp::<i32, 17>(&left, &right, &mut output).unwrap_err();
    let fault = recovered.arithmetic_fault().unwrap();
    assert!(fault.recovered && fault.invocation_id == 7);
    assert_eq!(output[7], i32::MAX);
    assert_eq!(output[17..], [77; 3]);
    left.fill(10);
    source::sub::<i32, 17>(&left, &right, &mut output)?;
    assert_eq!(output[..17], [8; 17]);
    global::clear_thread_cache()?;
    global::use_defaults()?;
    println!(
        "CPU checked source Add/Sub preserves fatal output, publishes recovered saturation, and retries with intact tails."
    );
    Ok(())
}

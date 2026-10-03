//! Actual annotated checked casts with useful Clamp output and fatal whole-call rollback.
#[path = "../../../cpu/tests/prepared_conversion/source/source.rs"]
#[allow(dead_code)]
mod source;
use pcu_facade::global;
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let input = [-0.0_f64, f64::MAX, 1.0];
    let mut output = [17.0_f32; 5];
    let error = source::narrow_clamp::<3>(&input, &mut output).unwrap_err();
    assert!(error.arithmetic_fault().unwrap().recovered);
    assert_eq!(
        output.map(f32::to_bits),
        [-0.0_f32, f32::MAX, 1.0, 17.0, 17.0].map(f32::to_bits)
    );
    let saved = output;
    let error = source::narrow_clamp::<3>(&[-0.0, f64::MAX, f64::NAN], &mut output).unwrap_err();
    assert!(!error.arithmetic_fault().unwrap().recovered);
    assert_eq!(output.map(f32::to_bits), saved.map(f32::to_bits));
    source::narrow::<3>(&[-0.0, 1.25, -2.5], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [-0.0_f32, 1.25, -2.5, 17.0, 17.0].map(f32::to_bits)
    );
    println!(
        "Vulkan checked cast: useful Clamp Err, fatal rollback, retry and signed zero preserved."
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

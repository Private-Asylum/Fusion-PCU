//! Ordinary mixed-width Clamp retains its complete useful output and observable range notice.
#[path = "../../tests/prepared_conversion/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use pcu_facade::{global,PcuExecutionFaultKind};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let mut output = [99_f32; 5];
    let notice = source::narrow_clamp::<3>(&[1.0, f64::MAX, -0.0], &mut output)
        .unwrap_err()
        .arithmetic_fault()
        .unwrap();
    assert_eq!(
        (notice.invocation_id, notice.kind, notice.recovered),
        (1, PcuExecutionFaultKind::ArithmeticOverflow, true)
    );
    assert_eq!(
        output.map(f32::to_bits),
        [1_f32, f32::MAX, -0.0, 99.0, 99.0].map(f32::to_bits)
    );
    let saved = output;
    assert!(
        !source::narrow_clamp::<3>(&[f64::MAX, 1.0, f64::NAN], &mut output)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap()
            .recovered
    );
    assert_eq!(output.map(f32::to_bits), saved.map(f32::to_bits));
    source::narrow_clamp::<3>(&[1.0, 2.0, 3.0], &mut output).unwrap();
    println!(
        "CPU clamped F64→F32 cast: saturated finite output+Err, later fatal rollback and retry"
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

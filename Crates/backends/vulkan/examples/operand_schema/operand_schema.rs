//! Repeated and independent-index reads use one actual input allocation.
#[rustfmt::skip]use pcu_facade::{pcu,global,PcuCheckedFloat};
#[pcu(invocations=3,crate_path=::pcu_facade)]
fn normalize<T: PcuCheckedFloat>(output: &mut [T], unused: &[T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] / input[0];
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let mut output = [17.0_f64; 5];
    normalize(&mut output, &[], &[2.0, 4.0, 8.0]).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [1.0_f64, 2.0, 4.0, 17.0, 17.0].map(f64::to_bits)
    );
    let saved = output;
    let fault = normalize(&mut output, &[], &[0.0, 4.0, 8.0])
        .unwrap_err()
        .arithmetic_fault()
        .unwrap();
    assert_eq!(fault.invocation_id, 0);
    assert!(!fault.recovered);
    assert_eq!(output.map(f64::to_bits), saved.map(f64::to_bits));
    normalize(&mut output, &[], &[4.0, 8.0, 16.0]).unwrap();
    assert_eq!(output.map(f64::to_bits), saved.map(f64::to_bits));
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    println!(
        "Vulkan uses one actual input, independent indexed/broadcast operands, and no storage for the unread declaration"
    );
}

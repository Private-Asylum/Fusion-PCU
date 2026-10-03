//! Requested bounded Portable map with explicit Vulkan provider selection.
#[rustfmt::skip]
use pcu_facade::{pcu,global,PcuF8E4M3FnBits,PcuCheckedFloat};
#[pcu(invocations=N,crate_path=::pcu_facade,flag(deterministic))]
fn quotient<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] / right[id];
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let one = PcuF8E4M3FnBits::from_bits(0x38);
    let two = PcuF8E4M3FnBits::from_bits(0x40);
    let sentinel = PcuF8E4M3FnBits::from_bits(17);
    let mut output = [sentinel; 10];
    quotient::<_, 7>(&[two; 7], &[one; 7], &mut output).unwrap();
    assert_eq!(output[..7], [two; 7]);
    assert_eq!(output[7..], [sentinel; 3]);
    let before = output;
    assert!(quotient::<_, 7>(&[two; 7], &[PcuF8E4M3FnBits::from_bits(0); 7], &mut output).is_err());
    assert_eq!(output, before);
    quotient::<_, 7>(&[two; 7], &[one; 7], &mut output).unwrap();
    println!("Vulkan Portable FP8 quotient preserved fatal rollback, retry and tails");
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

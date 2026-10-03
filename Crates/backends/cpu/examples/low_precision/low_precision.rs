//! Ordinary generic CPU source with exact format-preserving bits and transactional faults.
use pcu_facade::pcu;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedFloat,
    PcuF8E4M3FnBits,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn divide<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] / right[id];
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let left = [
        PcuF8E4M3FnBits::from_bits(0x40),
        PcuF8E4M3FnBits::from_bits(0x80),
    ];
    let one = PcuF8E4M3FnBits::from_bits(0x38);
    let mut output = [PcuF8E4M3FnBits::from_bits(0x7e); 3];
    divide::<_, 2>(&left, &[one; 2], &mut output).unwrap();
    assert_eq!(output.map(PcuF8E4M3FnBits::to_bits), [0x40, 0x80, 0x7e]);
    let before = output;
    assert!(divide::<_, 2>(&left, &[one, PcuF8E4M3FnBits::from_bits(0)], &mut output).is_err());
    assert_eq!(output, before);
    println!(
        "E4M3FN bits: {:x?}; failed division preserved output",
        output.map(PcuF8E4M3FnBits::to_bits)
    );
}

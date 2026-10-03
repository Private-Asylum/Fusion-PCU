//! Ordinary checked derivative owners preserve bit selection and previously published results.
#[path = "../../tests/low_tensor_backward/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use pcu_facade::{global,PcuF8E5M2Bits,PcuExecutionFaultKind};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let input = [
        PcuF8E5M2Bits::from_bits(0x3c),
        PcuF8E5M2Bits::from_bits(0xbc),
        PcuF8E5M2Bits::from_bits(0x80),
    ];
    let gradient = [PcuF8E5M2Bits::from_bits(0x80); 3];
    let previous = source::checked(&input[..], &gradient[..]).unwrap();
    let invalid = [PcuF8E5M2Bits::from_bits(0x7c); 3];
    let error = source::checked(&input[..], &invalid[..]).unwrap_err();
    assert_eq!(
        error.arithmetic_fault().unwrap().kind,
        PcuExecutionFaultKind::InvalidFloatingOperand
    );
    global::clear_thread_cache().unwrap();
    let mut readback = [PcuF8E5M2Bits::from_bits(0x3c); 5];
    previous.read_into(&mut readback).unwrap();
    assert_eq!(
        readback.map(PcuF8E5M2Bits::to_bits),
        [0x80, 0, 0, 0x3c, 0x3c]
    );
    source::checked(&input[..], &gradient[..])
        .unwrap()
        .read_into(&mut readback)
        .unwrap();
    println!(
        "CPU checked FP8 derivative: exact selected signed zero, inactive +0, fault-safe owners and retry"
    );
    global::use_defaults().unwrap();
}

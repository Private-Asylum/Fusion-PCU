//! Named byte-format owned results retain exact bytes and independent sibling lifetimes.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/low_tensor/source/source.rs"]
#[allow(dead_code)] // Other genuine source functions serve the canonical workload.
mod source;
use fusion_pcu::{PcuF8E4M3FnBits as F, PcuExecutionError, PcuExecutionFaultKind};
fn main() {
    let _device = selection::selected_device();
    let input = [F::from_bits(0x38); 65];
    let a = source::identity(input.as_slice()).unwrap();
    let prior = source::add(&a, &a).unwrap();
    let bad = [F::from_bits(0); 65];
    let result = source::div(&a, bad.as_slice());
    assert!(
        matches!(result,Err(PcuExecutionError::ArithmeticFault(f)) if f.kind==PcuExecutionFaultKind::DivideByZero&&!f.recovered)
    );
    let mut output = [F::from_bits(0); 67];
    prior.read_into(&mut output).unwrap();
    assert_eq!(&output[..65], &[F::from_bits(0x40); 65]);
    assert_eq!(&output[65..], &[F::from_bits(0); 2]);
    source::mul(&a, &a).unwrap().read_into(&mut output).unwrap();
    assert_eq!(&output[..65], &input);
    println!(
        "E4M3FN fresh exact owned output, retained sibling, zero-divisor failure and retry pass"
    );
}

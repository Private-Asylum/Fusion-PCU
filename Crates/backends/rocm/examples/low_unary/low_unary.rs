//! Genuine checked low unary source: recovery remains observable and output is published.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/low_unary/source/source.rs"]
#[allow(dead_code)]
mod source;
fn main() {
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    })
    .unwrap();
    let input = [fusion_pcu::PcuF16Bits::from_bits(1); 65];
    let mut output = [fusion_pcu::PcuF16Bits::from_bits(0x3c00); 67];
    let result = source::grid_neg::<fusion_pcu::PcuF16Bits, 65>(&input, &mut output);
    assert!(
        matches!(result,Err(fusion_pcu::PcuExecutionError::ArithmeticFault(f)) if f.recovered&&f.invocation_id==0)
    );
    assert_eq!(output[0].to_bits(), 0x8001);
    assert_eq!(output[65].to_bits(), 0x3c00);
    println!("Exact half Neg publishes recovered underflow and preserves tails");
}

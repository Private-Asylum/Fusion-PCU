//! Named FP8 derivative selects exact upstream storage and validates inactive upstream.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/checked_relu_backward/source/source.rs"]
#[allow(dead_code)] // Other source functions serve the canonical parity fixtures.
mod source;
use fusion_pcu::PcuF8E4M3FnBits as F;
fn main() {
    let _device = selection::selected_device();
    let x = [
        F::from_bits(0),
        F::from_bits(0x80),
        F::from_bits(0x7e),
        F::from_bits(0xb8),
    ];
    let dy = [
        F::from_bits(0x80),
        F::from_bits(0x80),
        F::from_bits(0x80),
        F::from_bits(1),
    ];
    let prior = source::checked(&x[..], &dy[..]).unwrap();
    let mut result = [F::from_bits(0x39); 6];
    prior.read_into(&mut result).unwrap();
    assert_eq!(
        &result[..4],
        &[
            F::from_bits(0),
            F::from_bits(0),
            F::from_bits(0x80),
            F::from_bits(0)
        ]
    );
    assert_eq!(&result[4..], &[F::from_bits(0x39); 2]);
    assert!(source::checked(&[F::from_bits(0xb8)][..], &[F::from_bits(0x7f)][..]).is_err());
    prior.read_into(&mut result).unwrap();
    println!(
        "E4M3FN finite maximum, upstream signed zero, inactive NaN failure and retained sibling pass"
    );
}

//! Actual ordinary composed FP8 checked map.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/composed_float_maps/source/source.rs"]
#[allow(dead_code)] // The canonical fixture qualifies every source shape.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuF8E4M3FnBits,
};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        ..Default::default()
    })
    .unwrap();
    let input = [PcuF8E4M3FnBits::from_bits(0x38); 65];
    let mut output = [PcuF8E4M3FnBits::from_bits(0x39); 67];
    source::map::grid::<_, 65>(&[], &mut output, &input).unwrap();
    assert!(output[..65].iter().all(|value| value.to_bits() == 0x40));
    assert!(output[65..].iter().all(|value| value.to_bits() == 0x39));
    println!("actual cuda #[pcu] composed E4M3FN (x+x)*x: exact2, two untouched tails");
}

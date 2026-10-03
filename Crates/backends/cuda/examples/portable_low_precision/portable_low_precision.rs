//! Explicit deterministic source over the proved low-format map; no whole-model claim.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/checked_low_precision/source/source.rs"]
#[allow(dead_code)]
mod source;
use fusion_pcu::PcuF16Bits;
fn main() {
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Cuda,
        ..Default::default()
    })
    .unwrap();
    let input = [PcuF16Bits::from_bits(0x3c00); 3];
    let mut output = [PcuF16Bits::from_bits(0); 4];
    source::mul::portable::<PcuF16Bits, 3>(&input, &input, &mut output).unwrap();
    assert_eq!(&output[..3], &input);
    assert_eq!(output[3].to_bits(), 0);
    println!("PortableV1 binary16 product={output:?}");
    fusion_pcu::global::clear_thread_cache().unwrap();
}

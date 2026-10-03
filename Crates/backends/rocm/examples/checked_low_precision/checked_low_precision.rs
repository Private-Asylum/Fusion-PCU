//! Ordinary generic checked arithmetic on explicitly named storage formats.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/checked_low_precision/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use fusion_pcu::{PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits};
fn main() {
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        ..Default::default()
    })
    .unwrap();
    macro_rules! demonstrate {
        ($ty:ty,$one:expr) => {{
            let input = [<$ty>::from_bits($one); 3];
            let mut output = [<$ty>::from_bits(0); 4];
            source::mul::direct::<$ty, 3>(&input, &input, &mut output).unwrap();
            assert_eq!(&output[..3], &input);
            assert_eq!(output[3].to_bits(), 0);
            println!("{}: {output:?}", stringify!($ty));
        }};
    }
    demonstrate!(PcuF16Bits, 0x3c00);
    demonstrate!(PcuBf16Bits, 0x3f80);
    demonstrate!(PcuF8E4M3FnBits, 0x38);
    demonstrate!(PcuF8E5M2Bits, 0x3c);
    fusion_pcu::global::clear_thread_cache().unwrap();
}

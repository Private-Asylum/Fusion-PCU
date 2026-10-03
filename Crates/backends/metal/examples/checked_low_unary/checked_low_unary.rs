//! Ordinary generic low unary source with exact byte publication and canonical signed zero.
#[path = "../../benches/checked_low_unary/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
fn main() {
    if !cfg!(target_os = "macos") {
        println!("SKIP: actual low unary Metal hardware required");
        return;
    }
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    macro_rules! run {
        ($ty:ty) => {{
            let input = [-1.0, -0.0, 2.0].map(|value| <$ty>::pcu_checked_from_f32(value).unwrap());
            let sentinel = <$ty>::pcu_checked_from_f32(91.0).unwrap();
            let mut output = [sentinel; 5];
            source::negate::<$ty, 3>(&input, &mut output).unwrap();
            source::relu::<$ty, 3>(&input, &mut output).unwrap();
            let expected = [0.0, 0.0, 2.0, 91.0, 91.0]
                .map(|value| <$ty>::pcu_checked_from_f32(value).unwrap());
            assert_eq!(
                output.map(|value| value.to_bits()),
                expected.map(|value| value.to_bits())
            );
            let least = <$ty>::from_bits(1);
            let recovered =
                source::relu_tight_clamp::<$ty, 3>(&[least, input[0], least], &mut output)
                    .unwrap_err();
            assert!(recovered.recovered_range_fault().is_some());
            assert_eq!(
                output.map(|value| value.to_bits()),
                [
                    least.to_bits(),
                    0,
                    least.to_bits(),
                    sentinel.to_bits(),
                    sentinel.to_bits()
                ]
            );
            println!(
                "{} exact unary bits: {:?}",
                stringify!($ty),
                output.map(|value| value.to_bits())
            );
        }};
    }
    run!(PcuF16Bits);
    run!(PcuBf16Bits);
    run!(PcuF8E4M3FnBits);
    run!(PcuF8E5M2Bits);
}

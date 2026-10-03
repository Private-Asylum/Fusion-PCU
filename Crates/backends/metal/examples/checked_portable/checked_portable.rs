//! Actual ordinary checked source across four integer-realized low-precision encodings.
#[path = "../../benches/checked_portable/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
macro_rules! run {
    ($ty:ty) => {{
        let left = [1.0, 2.0, 3.0].map(|value| <$ty>::pcu_checked_from_f32(value).unwrap());
        let right = [2.0, 4.0, 6.0].map(|value| <$ty>::pcu_checked_from_f32(value).unwrap());
        let sentinel = <$ty>::pcu_checked_from_f32(91.0).unwrap();
        let mut output = [sentinel; 5];
        source::add::<$ty, 3>(&left, &right, &mut output).unwrap();
        source::sub::<$ty, 3>(&left, &right, &mut output).unwrap();
        source::mul::<$ty, 3>(&left, &right, &mut output).unwrap();
        source::div::<$ty, 3>(&left, &right, &mut output).unwrap();
        let half = <$ty>::pcu_checked_from_f32(0.5).unwrap();
        assert_eq!(
            output.map(|value| value.to_bits()),
            [half, half, half, sentinel, sentinel].map(|value| value.to_bits())
        );
        println!(
            "{} checked integer GPU source bits: {:?}",
            stringify!($ty),
            output.map(|value| value.to_bits())
        );
    }};
}
fn main() {
    if !cfg!(target_os = "macos") {
        println!("SKIP: actual checked low-format Metal requires hardware");
        return;
    }
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    run!(PcuF16Bits);
    run!(PcuBf16Bits);
    run!(PcuF8E4M3FnBits);
    run!(PcuF8E5M2Bits);
}

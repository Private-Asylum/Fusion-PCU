//! Genuine CPU owned source routing, checked training and independent policy rejection.
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    global::PcuBackendChoice,
    global::PcuExecutionPolicy,
};
#[path = "../prepared_tensor/allocation/allocation.rs"]
mod allocation;
#[path = "../../benches/checked_tensor/source/source.rs"]
mod source;

#[test]
fn explicit_cpu_source_loss_training_and_policy_boundaries() {
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Cpu,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    macro_rules! check {
        ($ty:ty) => {{
            let prediction: [$ty; 2] = [1.0, 3.0];
            let target: [$ty; 2] = [0.0, 1.0];
            let mut output = [99.0 as $ty; 2];
            let loss = source::loss(&prediction, &target).unwrap();
            loss.read_into(&mut output).unwrap();
            assert_eq!(
                output.map(<$ty>::to_bits),
                [2.5 as $ty, 99.0 as $ty].map(<$ty>::to_bits)
            );
            let owned = source::identity(&prediction).unwrap();
            let next = source::loss::<$ty, 2>(&owned, &target).unwrap();
            next.read_into(&mut output).unwrap();
            assert_eq!(output[0].to_bits(), (2.5 as $ty).to_bits());
            assert_eq!(
                allocation::count(|| {
                    let output_owner = source::loss(&prediction, &target).unwrap();
                    output_owner.read_into(&mut output).unwrap();
                    drop(output_owner);
                }),
                1,
                "warm source loss must allocate only escaping output data"
            );
            assert!(source::boundary_loss(&prediction, &target).is_err());
            source::native_loss(&prediction, &target)
                .unwrap()
                .read_into(&mut output)
                .unwrap();
            assert_eq!(output[0].to_bits(), (2.5 as $ty).to_bits());
            assert!(source::portable_loss(&prediction, &target).is_err());
            source::precision_loss(&prediction, &target)
                .unwrap()
                .read_into(&mut output)
                .unwrap();
            assert_eq!(output[0].to_bits(), (2.5 as $ty).to_bits());
            assert!(source::loss(&[<$ty>::NAN, 3.0], &target).is_err());
            source::loss(&prediction, &target).unwrap();
            let x: [[$ty; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
            let xt = [[1.0, 3.0], [2.0, 4.0]];
            let w = [[0.5], [0.25]];
            let y = [[0.0], [1.0]];
            let trained = source::training(&x, &xt, &w, &y).unwrap();
            trained.read_into(&mut output).unwrap();
            assert_eq!(
                output.map(<$ty>::to_bits),
                [-2.25 as $ty, -3.75 as $ty].map(<$ty>::to_bits)
            );
            assert_eq!(
                allocation::count(|| {
                    let output_owner = source::training(&x, &xt, &w, &y).unwrap();
                    output_owner.read_into(&mut output).unwrap();
                    drop(output_owner);
                }),
                1,
                "warm source training must allocate only escaping output data"
            );
            assert!(source::training(&x, &xt, &w, &[[<$ty>::MAX], [0.0]]).is_err());
        }};
    }
    check!(f32);
    check!(f64);
    assert!(source::identity(&[fusion_pcu::PcuF16Bits::from_bits(0x3c00); 2]).is_ok());
    assert!(source::identity(&[fusion_pcu::PcuBf16Bits::from_bits(0x3f80); 2]).is_ok());
}

//! Static Session selection must retain composed effects and joint host publication.
#[rustfmt::skip]
use crate::{
    MlxError,
    MlxRuntime,
};
use super::source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFaultKind,
    PcuHostDispatchError,
};
#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires authentic Mlx host composition"
)]
fn session_aggregate_keeps_ten_integer_two_float_ordered_host_publication() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let backend = &session;
    macro_rules! integer { ($($ty:ty),*) => { $(
        let mut run = source::integer_prepare::<$ty, 7, _>(backend).unwrap();
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 9] = core::array::from_fn(|index|
                <$ty>::try_from(1_u8 + u8::try_from(index).unwrap() + phase).unwrap());
            let seed = <$ty>::try_from(2_u8).unwrap();
            let sentinel = <$ty>::try_from(19_u8).unwrap();
            let mut stage = [sentinel; 10];
            let mut output = [sentinel; 11];
            run(&mut [], &input, &seed, &mut stage, &mut output).unwrap();
            assert_eq!(&stage[..7], &input[..7]);
            assert_eq!(&stage[7..], &[sentinel; 3]);
            let expected: [$ty; 7] = core::array::from_fn(|lane|
                (input[lane] + 2) * input[lane] - input[lane]);
            assert_eq!(&output[..7], &expected);
            assert_eq!(&output[7..], &[sentinel; 4]);
            let mut short = [sentinel; 6];
            let old_stage = stage;
            assert!(matches!(run(&mut [], &input, &seed, &mut stage, &mut short),
                Err(PcuHostDispatchError::BufferTooSmall(_))));
            assert_eq!(stage, old_stage);
            assert_eq!(short, [sentinel; 6]);
        }
    )* }; }
    integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
    macro_rules! floating { ($($ty:ty),*) => { $(
        let mut run = source::floating_prepare::<$ty, 7, _>(backend).unwrap();
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 9] = core::array::from_fn(|index|
                <$ty>::from(1_u8 + u8::try_from(index).unwrap() + phase));
            let seed = <$ty>::from(2_u8);
            let sentinel = <$ty>::from(19_u8);
            let mut stage = [sentinel; 10];
            let mut output = [sentinel; 11];
            run(&mut [], &input, &seed, &mut stage, &mut output).unwrap();
            assert_eq!(&stage.map(<$ty>::to_bits)[..7], &input.map(<$ty>::to_bits)[..7]);
            assert_eq!(&stage.map(<$ty>::to_bits)[7..], &[sentinel.to_bits(); 3]);
            let expected: [$ty; 7] = core::array::from_fn(|lane| {
                let value = u16::from(1_u8 + u8::try_from(lane).unwrap() + phase);
                <$ty>::from((value + 2) * value + value)
            });
            assert_eq!(&output.map(<$ty>::to_bits)[..7], &expected.map(<$ty>::to_bits));
            assert_eq!(&output.map(<$ty>::to_bits)[7..], &[sentinel.to_bits(); 4]);
            let old_stage = stage;
            let old_output = output;
            assert!(matches!(run(&mut [], &input, &0.0, &mut stage, &mut output),
                Err(PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)))
                if fault.kind == PcuExecutionFaultKind::DivideByZero && !fault.recovered));
            assert_eq!(stage.map(<$ty>::to_bits), old_stage.map(<$ty>::to_bits));
            assert_eq!(output.map(<$ty>::to_bits), old_output.map(<$ty>::to_bits));
            run(&mut [], &input, &seed, &mut stage, &mut output).unwrap();
            assert_eq!(&output.map(<$ty>::to_bits)[..7], &expected.map(<$ty>::to_bits));
        }
    )* }; }
    floating!(f32, f64);
}

#[path = "mixed/mixed.rs"]
mod mixed;

#[path = "fault/fault.rs"]
mod fault;

//! Six-format ordinary borrowed encoded owners and transactional unary publication.
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedFloat,
    PcuTensor,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[rustfmt::skip]
use super::{
    source,
    Policy,
};
trait Sample: PcuCheckedFloat {
    const SIGN: u64;
    fn bits(bits: u64) -> Self;
    fn value(value: f32) -> Self;
}
macro_rules! sample {
    ($ty:ty,$word:ty,$sign:expr,$max:expr,$convert:expr) => {
        impl Sample for $ty {
            const SIGN: u64 = $sign;
            fn bits(bits: u64) -> Self {
                Self::from_bits(<$word>::try_from(bits).unwrap())
            }
            fn value(value: f32) -> Self {
                ($convert)(value)
            }
        }
    };
}
sample!(PcuF16Bits, u16, 0x8000, 0x7bff, |v| {
    PcuF16Bits::pcu_checked_from_f32(v).unwrap()
});
sample!(PcuBf16Bits, u16, 0x8000, 0x7f7f, |v| {
    PcuBf16Bits::pcu_checked_from_f32(v).unwrap()
});
sample!(PcuF8E4M3FnBits, u8, 0x80, 0x7e, |v| {
    PcuF8E4M3FnBits::pcu_checked_from_f32(v).unwrap()
});
sample!(PcuF8E5M2Bits, u8, 0x80, 0x7b, |v| {
    PcuF8E5M2Bits::pcu_checked_from_f32(v).unwrap()
});
sample!(f32, u32, 0x8000_0000, 0x7f7f_ffff, |v| v);
sample!(
    f64,
    u64,
    0x8000_0000_0000_0000,
    0x7fef_ffff_ffff_ffff,
    f64::from
);
fn verify<T: Sample>(owner: &PcuTensor<T>, values: [T; 3]) {
    let mut output = [T::value(6.0); 7];
    owner.read_into(&mut output).unwrap();
    for (actual, wanted) in output.iter().zip(
        values
            .into_iter()
            .chain([T::value(1.0); 2])
            .chain([T::value(6.0); 2]),
    ) {
        assert_eq!(actual.encode_le().as_ref(), wanted.encode_le().as_ref());
    }
}
fn case<T: Sample>() {
    let values = [T::value(-2.0), T::bits(T::SIGN), T::value(4.0)];
    let input = source::retain(&values).unwrap();
    let mut output = source::retain(&[T::value(1.0); 5]).unwrap();
    source::negate::<T, 3>(&input, &mut output).unwrap();
    verify(&output, [T::value(2.0), T::bits(0), T::value(-4.0)]);
    source::relu::<T, 3>(&input, &mut output).unwrap();
    verify(&output, [T::bits(0), T::bits(0), T::value(4.0)]);
    let tiny = [T::bits(1), T::bits(T::SIGN | 1), T::bits(T::SIGN)];
    let tiny_owner = source::retain(&tiny).unwrap();
    let error = source::negate_tight_clamp::<T, 3>(&tiny_owner, &mut output);
    assert!(
        matches!(error,Err(PcuExecutionError::ArithmeticFault(fault)) if fault.recovered && fault.invocation_id==0 && fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    verify(&output, [T::bits(T::SIGN | 1), T::bits(1), T::bits(0)]);
    let fatal = [T::bits(1), T::bits(T::SIGN - 1), T::bits(0)];
    assert!(
        matches!(source::negate_tight_clamp::<T,3>(&fatal,&mut output),Err(PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id==1)
    );
    verify(&output, [T::bits(T::SIGN | 1), T::bits(1), T::bits(0)]);
    let mut host = [T::value(1.0); 5];
    source::relu_gradual::<T, 3>(&tiny_owner, &mut host).unwrap();
    for (actual, wanted) in host.iter().zip([
        tiny[0],
        T::bits(0),
        T::bits(0),
        T::value(1.0),
        T::value(1.0),
    ]) {
        assert_eq!(actual.encode_le().as_ref(), wanted.encode_le().as_ref());
    }
    source::negate::<T, 3>(&values, &mut output).unwrap();
    drop(input);
    drop(tiny_owner);
    global::clear_thread_cache().unwrap();
    verify(&output, [T::value(2.0), T::bits(0), T::value(-4.0)]);
}
fn width<T: Sample>() {
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    Policy::IeeeAfterRounding,
                    Policy::RejectSubnormalResult,
                    Policy::AllowGradualUnderflow,
                ] {
                    let mut policy = global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Mlx,
                        numerical_mode,
                        float_underflow,
                        ..global::PcuExecutionPolicy::default()
                    };
                    policy.numerical_options.compound_arithmetic = compound;
                    policy.numerical_options.precision = precision;
                    global::configure(policy).unwrap();
                    case::<T>();
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual MLX six-format unary encoded owner borrows, immutable prefix publication and exact terminal statuses."]
fn ordinary_six_format_unary_resident_publication() {
    width::<PcuF16Bits>();
    width::<PcuBf16Bits>();
    width::<PcuF8E4M3FnBits>();
    width::<PcuF8E5M2Bits>();
    width::<f32>();
    width::<f64>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[path = "offers/offers.rs"]
mod offers;

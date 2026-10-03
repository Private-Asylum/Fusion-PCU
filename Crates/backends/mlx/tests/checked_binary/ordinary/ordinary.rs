//! Actual ordinary six-format encoded owner borrows, mixed inputs and private publication.
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuTensor,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuExecutionError,
    PcuExecutionFaultKind,
};
#[rustfmt::skip]
use super::{
    Sample,
    Policy,
    source,
};
fn verify<T: Sample>(owner: &PcuTensor<T>, prefix: [T; 3], tail: T) {
    let mut output = [T::value(6.0); 7];
    owner.read_into(&mut output).unwrap();
    for (actual, wanted) in output.iter().zip(
        prefix
            .into_iter()
            .chain([tail; 2])
            .chain([T::value(6.0); 2]),
    ) {
        assert_eq!(actual.encode_le().as_ref(), wanted.encode_le().as_ref());
    }
}
fn case<T: Sample>() {
    let left = [T::value(2.0); 3];
    let right = [T::value(4.0); 3];
    let lhs = source::retain(&left).unwrap();
    let rhs = source::retain(&right).unwrap();
    let mut output = source::retain(&[T::value(1.0); 5]).unwrap();
    source::add::<T, 3>(&lhs, &right, &mut output).unwrap();
    verify(&output, [T::value(6.0); 3], T::value(1.0));
    source::sub::<T, 3>(&lhs, &rhs, &mut output).unwrap();
    verify(&output, [T::value(-2.0); 3], T::value(1.0));
    source::swapped::<T, 3>(&mut output, &rhs, &left).unwrap();
    verify(&output, [T::value(-2.0); 3], T::value(1.0));
    source::repeated::<T, 3>(&lhs, &[], &mut output).unwrap();
    verify(&output, [T::value(4.0); 3], T::value(1.0));
    source::grid::<T, 3>(&lhs, &rhs, &mut output).unwrap();
    verify(&output, [T::value(0.5); 3], T::value(1.0));
    let error = source::div::<T, 3>(&lhs, &[T::value(0.0), right[0], T::value(0.0)], &mut output);
    assert!(
        matches!(error, Err(PcuExecutionError::ArithmeticFault(fault))
        if !fault.recovered && fault.invocation_id == 0 && fault.kind == PcuExecutionFaultKind::DivideByZero)
    );
    verify(&output, [T::value(0.5); 3], T::value(1.0));
    let tiny = [T::from_bits(1), T::from_bits(T::MAX), left[0]];
    let rhs_values = [T::value(0.0), T::from_bits(T::MAX), right[0]];
    let error = source::add_tight_clamp::<T, 3>(&tiny, &rhs_values, &mut output);
    assert!(
        matches!(error, Err(PcuExecutionError::ArithmeticFault(fault))
        if fault.recovered && fault.invocation_id == 0 && fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    verify(&output, [tiny[0], tiny[1], T::value(6.0)], T::value(1.0));
    let fatal = [tiny[0], T::from_bits(T::SIGN - 1), left[0]];
    assert!(
        matches!(source::add_tight_clamp::<T, 3>(&fatal, &rhs_values, &mut output),
        Err(PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id == 1)
    );
    verify(&output, [tiny[0], tiny[1], T::value(6.0)], T::value(1.0));
    source::mul::<T, 3>(&lhs, &rhs, &mut output).unwrap();
    drop(lhs);
    drop(rhs);
    global::clear_thread_cache().unwrap();
    verify(&output, [T::value(8.0); 3], T::value(1.0));
}
fn width<T: Sample>() {
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
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
                    global::configure(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Mlx,
                        numerical_mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            reproducibility: PcuReproducibility::Unspecified,
                        },
                        float_underflow,
                        ..Default::default()
                    })
                    .unwrap();
                    case::<T>();
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual ordinary MLX six-format host/resident/mixed binary publication."]
fn ordinary_six_format_binary_resident_publication() {
    width::<super::PcuF16Bits>();
    width::<super::PcuBf16Bits>();
    width::<super::PcuF8E4M3FnBits>();
    width::<super::PcuF8E5M2Bits>();
    width::<f32>();
    width::<f64>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

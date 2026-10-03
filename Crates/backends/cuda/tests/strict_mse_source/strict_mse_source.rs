//! Actual source ordered loss, dependent forward loss, faults and successful retry.
extern crate pcu_facade as fusion_pcu;
#[allow(dead_code)] // Shared exact width construction also serves benchmark-only domains.
#[path = "../../benches/relu_backward/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/strict_mse/source/source.rs"]
mod source;
use oracle::Scalar;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuTensor,
};
fn verify<T: Scalar>(output: &PcuTensor<T>, expected: T) {
    assert!(output.shape().is_empty());
    let mut actual = [T::default()];
    output.read_into(&mut actual).unwrap();
    oracle::verify(&[expected], &actual);
}
fn execute<T: Scalar>(options: fusion_pcu::PcuNumericalOptions) {
    let zero = T::default();
    let one = T::from_units(1, 1);
    let two = T::from_units(2, 1);
    for _ in 0..3 {
        verify(
            &source::strict(&[one, two], &[zero, zero]).unwrap(),
            T::from_units(5, 2),
        );
        verify(
            &source::forward_loss(&[T::from_units(-1, 1), two], &[zero, one]).unwrap(),
            T::from_units(1, 2),
        );
    }
    let input = [[one, zero], [zero, one]];
    let weights = [[two], [T::from_units(-1, 1)]];
    let target = [[one], [zero]];
    let updated = source::training_step(&input, &input, &weights, &target).unwrap();
    let mut actual = [zero; 2];
    updated.read_into(&mut actual).unwrap();
    oracle::verify(&[T::from_units(3, 2), T::from_units(-1, 1)], &actual);
    let fault =
        source::training_step(&input, &input, &weights, &[[T::infinity()], [zero]]).unwrap_err();
    assert!(format!("{fault:?}").contains("Subtract"));
    let _retry = source::training_step(&input, &input, &weights, &target).unwrap();
    for (prediction, expected) in [
        (T::infinity(), "Subtract"),
        (T::max(), "Multiply"),
        (T::tiny(), "Multiply"),
    ] {
        let error = source::strict(&[one, prediction], &[zero, zero]).unwrap_err();
        let text = format!("{error:?}");
        assert!(
            text.contains(expected) && text.contains("reduction_index: 1"),
            "{text}"
        );
        verify(&source::strict(&[one], &[zero]).unwrap(), one);
    }
    verify(&source::gradual(&[T::tiny()], &[zero]).unwrap(), zero);
    assert!(source::tight(&[T::tiny()], &[zero]).is_err());
    verify(&source::optimized(&[one], &[zero]).unwrap(), one);
    if options.compound_arithmetic == fusion_pcu::PcuCompoundArithmeticPolicy::Checked {
        let error = source::boundary(&[one], &[zero]).unwrap_err();
        assert!(format!("{error:?}").contains("Unsupported"));
    }
    let error = source::portable(&[one], &[zero]).unwrap_err();
    assert!(format!("{error:?}").contains("Unsupported"));
}
#[test]
#[ignore = "requires authorized GPU; run serially after activity inspection"]
fn actual_source_strict_mse_both_widths_ordered_faults_forward_loss_and_retry() {
    for compound_arithmetic in [
        fusion_pcu::PcuCompoundArithmeticPolicy::Checked,
        fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            fusion_pcu::PcuPrecisionPolicy::Preserve,
            fusion_pcu::PcuPrecisionPolicy::BackendOptimized,
        ] {
            let options = fusion_pcu::PcuNumericalOptions {
                compound_arithmetic,
                precision,
                ..Default::default()
            };
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Cuda,
                device: Some(0),
                block_size: 256,
                numerical_options: options,
                ..Default::default()
            })
            .unwrap();
            global::clear_thread_cache().unwrap();
            execute::<f32>(options);
            execute::<f64>(options);
            // Exact subnormal square followed by inexact tiny mean must fault at the final division.
            let error =
                source::strict(&[f32::from_bits(53 << 23), 0.0, 0.0], &[0.0; 3]).unwrap_err();
            assert!(format!("{error:?}").contains("Divide"), "{error:?}");
            global::clear_thread_cache().unwrap();
        }
    }
    global::use_defaults().unwrap();
}

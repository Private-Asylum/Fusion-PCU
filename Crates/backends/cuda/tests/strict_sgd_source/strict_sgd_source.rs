//! Authored checked optimizer acceptance, source residency and transactional failure.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_sgd/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/strict_sgd/source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuTensor,
};
use oracle::Scalar;
fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        block_size: 256,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
fn verify<T: Scalar>(output: &PcuTensor<T>, expected: &[T]) {
    let mut actual = vec![T::default(); expected.len()];
    output.read_into(&mut actual).unwrap();
    oracle::verify(expected, &actual);
}
fn success<T: Scalar>() {
    eprintln!("source width {}", T::LABEL);
    for phase in 0..3 {
        let (weights, gradient, expected) = oracle::inputs::<T, 17>(phase);
        let output = source::update(&weights, &gradient).unwrap();
        verify(&output, &expected);
        let w = source::identity(&weights).unwrap();
        let g = source::identity(&gradient).unwrap();
        let escaped = source::update::<T, 17>(&w, &g).unwrap();
        verify(&w, weights.as_slice());
        verify(&g, gradient.as_slice());
        drop(w);
        drop(g);
        global::clear_thread_cache().unwrap();
        verify(&escaped, &expected);
        let half = T::from_units(1, 2);
        let chain_expected: Vec<_> = weights
            .iter()
            .zip(gradient.iter())
            .map(|(&w, &g)| {
                w.pcu_checked_sub(half.pcu_checked_mul(g).unwrap())
                    .unwrap()
                    .pcu_checked_sub(half.pcu_checked_mul(g).unwrap())
                    .unwrap()
            })
            .collect();
        verify(
            &source::chain(&weights, &gradient).unwrap(),
            &chain_expected,
        );
    }
}
fn faults<T: Scalar>() {
    let one = T::from_units(1, 1);
    let zero = T::default();
    let cases = [
        (
            source::update(&[one], &[T::tiny()]).unwrap_err(),
            "Multiply",
            "ArithmeticUnderflow",
        ),
        (
            source::overflow(&[one], &[T::max()]).unwrap_err(),
            "Multiply",
            "ArithmeticOverflow",
        ),
        (
            source::update(&[T::max()], &[zero.pcu_checked_sub(T::max()).unwrap()]).unwrap_err(),
            "Subtract",
            "ArithmeticOverflow",
        ),
        (
            source::chain(&[T::infinity()], &[one]).unwrap_err(),
            "Subtract",
            "InvalidFloatingOperand",
        ),
    ];
    for (error, step, kind) in cases {
        let text = format!("{error:?}");
        assert!(
            text.contains(step)
                && text.contains(kind)
                && text.contains("element_index: 0")
                && text.contains("reduction_index: 0"),
            "{text}"
        );
        verify(
            &source::update(&[one], &[one]).unwrap(),
            &[T::from_units(1, 2)],
        );
    }
    // Exact tiny results are legal under IEEE; inexact tiny-after-rounding faults, gradual permits.
    verify(
        &source::update(&[T::tiny()], &[zero]).unwrap(),
        &[T::tiny()],
    );
    assert!(source::tight(&[T::tiny()], &[zero]).is_err());
    verify(&source::gradual(&[one], &[T::tiny()]).unwrap(), &[one]);
    let w = source::identity(&[one, one]).unwrap();
    let g = source::identity(&[T::tiny(), one]).unwrap();
    assert!(source::update::<T, 2>(&w, &g).is_err());
    verify(&w, &[one, one]);
    verify(&g, &[T::tiny(), one]);
    // First element's subtract fault wins over a later multiply fault.
    let error = source::update(&[T::infinity(), one], &[one, T::infinity()]).unwrap_err();
    assert!(format!("{error:?}").contains("element_index: 0"));
    verify(
        &source::witness(&[one], &[T::witness_gradient()]).unwrap(),
        &[zero],
    );
    let _ = source::zero(&[one], &[one]).unwrap();
    for error in [
        source::boundary(&[one], &[one]).unwrap_err(),
        source::native_strict(&[one], &[one]).unwrap_err(),
        source::portable(&[one], &[one]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"));
    }
}
#[test]
#[ignore = "requires authorized idle CUDA hardware; run serially"]
fn actual_source_strict_sgd_both_widths_changing_inputs_owned_outputs_and_chains() {
    configure();
    success::<f32>();
    success::<f64>();
    global::clear_thread_cache().unwrap();
}
#[test]
#[ignore = "requires authorized idle CUDA hardware; run serially"]
fn actual_source_strict_sgd_fault_boundaries_policies_owner_survival_and_retry() {
    configure();
    faults::<f32>();
    faults::<f64>();
    let g = [f32::from_bits(0x3f7f_fffe)];
    verify(&source::witness(&[1.0], &g).unwrap(), &[0.0]);
    global::clear_thread_cache().unwrap();
}

//! Source execution, exact bits, finite faults, discarded outputs and retry.
extern crate pcu_facade as fusion_pcu;
#[allow(dead_code)] // Shared width helpers include benchmark-only domain constructors.
#[path = "../../benches/relu_backward/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/relu_backward/source/source.rs"]
mod source;
use oracle::Scalar;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuTensor,
};
fn verify<T: Scalar>(output: &PcuTensor<T>, expected: &[T]) {
    let mut actual = vec![T::default(); expected.len()];
    output.read_into(&mut actual).unwrap();
    oracle::verify(expected, &actual);
}
fn execute<T: Scalar>() {
    let zero = T::default();
    let one = T::from_units(1, 1);
    let negative = T::from_units(-1, 1);
    let half = T::from_units(1, 2);
    let input = [zero, negative, one, one, one];
    let upstream = [one, one, negative, zero, T::tiny()];
    let expected = [zero, zero, negative, zero, T::tiny()];
    verify(&source::checked(&input, &upstream).unwrap(), &expected);
    verify(&source::strict(&input, &upstream).unwrap(), &expected);
    verify(&source::native(&input, &upstream).unwrap(), &expected);
    assert!(source::tight(&input, &upstream).is_err());
    // The inactive upstream is finite-checked. Fault must precede optimizer use/publication.
    for input in [one, negative, zero] {
        let error = source::checked(&[input], &[T::infinity()]).unwrap_err();
        assert!(format!("{error:?}").contains("InvalidFloatingOperand"));
        assert!(source::optimizer(&[one], &[input], &[T::infinity()]).is_err());
        verify(&source::checked(&[one], &[half]).unwrap(), &[half]);
    }
    let error = source::checked(&[T::infinity(), one], &[one, T::infinity()]).unwrap_err();
    assert!(format!("{error:?}").contains("element_index: 0"));
    verify(&source::tight(&[negative], &[T::tiny()]).unwrap(), &[zero]);
    verify(
        &source::optimizer(&[one, one], &[negative, one], &[one, one]).unwrap(),
        &[one, half],
    );
    verify(&source::strict_native(&[one], &[half]).unwrap(), &[half]);
    for input in [one, negative, zero] {
        let error = source::strict_native(&[input], &[T::infinity()]).unwrap_err();
        assert!(format!("{error:?}").contains("InvalidFloatingOperand"));
    }
    assert!(format!("{:?}", source::portable(&[one], &[one]).unwrap_err()).contains("Unsupported"));
}
#[test]
#[ignore = "requires authorized GPU; run serially after activity inspection"]
fn actual_source_relu_backward_checked_strict_native_fault_retry_and_optimizer() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        block_size: 256,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    execute::<f32>();
    execute::<f64>();
    let output = source::strict(&[-0.0_f32, 0.0, 1.0], &[-0.0_f32, -0.0, -0.0]).unwrap();
    verify(&output, &[0.0, 0.0, -0.0]);
    global::clear_thread_cache().unwrap();
    verify(&output, &[0.0, 0.0, -0.0]);
}

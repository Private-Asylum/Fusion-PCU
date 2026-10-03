//! Genuine F64 native loss: integer oracle, precise difference witness and ownership/policy proof.
extern crate pcu_facade as fusion_pcu;
#[allow(dead_code)] // Shared oracle includes benchmark-only edge constructors.
#[path = "../../benches/native_mse_f64/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/native_mse_f64/source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuTensor,
};
fn verify(output: &PcuTensor<f64>, expected: &[f64]) {
    assert!(output.shape().is_empty());
    let mut actual = [0.0_f64];
    output.read_into(&mut actual).unwrap();
    oracle::verify(expected, &actual);
}
fn verify_owner(output: &PcuTensor<f64>, expected: &[f64]) {
    let mut actual = vec![0.0; expected.len()];
    output.read_into(&mut actual).unwrap();
    oracle::verify(expected, &actual);
}
#[test]
#[ignore = "requires authorized idle GPU with BLAS; run serially"]
fn actual_source_native_f64_loss_width_policy_and_ownership() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        block_size: 256,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    for optimized in [false, true] {
        for phase in 0..3 {
            let (p, t, expected) = oracle::inputs::<f64, 65>(phase);
            verify(&source::loss(&p, &t, optimized).unwrap(), &expected);
            let p_owner = source::identity(&p).unwrap();
            let t_owner = source::identity(&t).unwrap();
            let escaped = if optimized {
                source::optimized::<f64, 65>(&p_owner, &t_owner).unwrap()
            } else {
                source::preserve::<f64, 65>(&p_owner, &t_owner).unwrap()
            };
            verify_owner(&p_owner, p.as_slice());
            verify_owner(&t_owner, t.as_slice());
            drop(p_owner);
            drop(t_owner);
            global::clear_thread_cache().unwrap();
            verify(&escaped, &expected);
        }
        // (1+2^-52)-1 is exactly2^-52, and its square is exactly2^-104.
        // F32 storage or a truncated scratch/kernel/BLAS ABI would erase this witness.
        let p = [f64::from_bits(0x3ff0_0000_0000_0001)];
        let expected = [f64::from_bits((1023 - 104) << 52)];
        verify(&source::loss(&p, &[1.0], optimized).unwrap(), &expected);
        verify(&source::loss(&[-0.0], &[0.0], optimized).unwrap(), &[0.0]);
        let exceptional = source::loss(&[f64::INFINITY], &[0.0], optimized).unwrap();
        let mut actual = [0.0_f64];
        exceptional.read_into(&mut actual).unwrap();
        assert!(actual[0].is_infinite());
        let nan = source::loss(&[f64::INFINITY], &[f64::INFINITY], optimized).unwrap();
        nan.read_into(&mut actual).unwrap();
        assert!(actual[0].is_nan());
        verify(&source::loss(&[2.0], &[1.0], optimized).unwrap(), &[1.0]);
    }
    for error in [
        source::checked(&[1.0_f64], &[1.0]).unwrap_err(),
        source::strict(&[1.0_f64], &[1.0]).unwrap_err(),
        source::portable(&[1.0_f64], &[1.0]).unwrap_err(),
        source::tight(&[1.0_f64], &[1.0]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"));
    }
    assert!(source::preserve::<f64, 0>(&[], &[]).is_err());
    let p = source::identity(&[1.0_f64, 2.0]).unwrap();
    let t = source::identity(&[1.0_f64, 2.0, 3.0]).unwrap();
    assert!(source::preserve::<f64, 2>(&p, &t).is_err());
    verify_owner(&p, &[1.0, 2.0]);
    verify_owner(&t, &[1.0, 2.0, 3.0]);
    global::clear_thread_cache().unwrap();
}

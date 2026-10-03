//! Genuine native F64 authoring, independent cancellation witness and policy/ownership proof.
extern crate pcu_facade as fusion_pcu;
#[allow(dead_code)] // Benchmark domain builders also serve this independent full-output proof.
#[path = "../../benches/native_sgd_f64/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/native_sgd_f64/source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{global, PcuTensor};
fn verify(output: &PcuTensor<f64>, expected: &[f64]) {
    assert_eq!(output.shape(), [expected.len()]);
    let mut observed = vec![0.0_f64; expected.len()];
    output.read_into(&mut observed).unwrap();
    oracle::verify(expected, &observed);
}
#[test]
#[ignore = "requires authorized idle GPU; run serially"]
fn actual_source_native_f64_frozen_rate_contraction_policy_and_ownership() {
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
            let (w, g, expected) = oracle::inputs::<f64, 65>(phase);
            verify(&source::update(&w, &g, optimized).unwrap(), &expected);
            let w_owner = source::identity(&w).unwrap();
            let g_owner = source::identity(&g).unwrap();
            let escaped = if optimized {
                source::optimized::<f64, 65>(&w_owner, &g_owner).unwrap()
            } else {
                source::preserve::<f64, 65>(&w_owner, &g_owner).unwrap()
            };
            verify(&w_owner, w.as_slice());
            verify(&g_owner, g.as_slice());
            drop(w_owner);
            drop(g_owner);
            global::clear_thread_cache().unwrap();
            verify(&escaped, &expected);
        }
    }
    // r=1+2^-23, g=1-2^-23+2^-46: exact product1+2^-69.
    // Separate F64 product rounds to1; explicitly fused subtraction returns -2^-69.
    let gradient = [f64::from_bits(0x3fef_ffff_c000_0080)];
    verify(
        &source::witness_preserve(&[1.0], &gradient).unwrap(),
        &[0.0],
    );
    verify(
        &source::witness_optimized(&[1.0], &gradient).unwrap(),
        &[f64::from_bits((1_u64 << 63) | ((1023 - 69) << 52))],
    );
    verify(&source::positive_zero(&[-0.0], &[1.0]).unwrap(), &[-0.0]);
    verify(&source::negative_zero(&[-0.0], &[1.0]).unwrap(), &[0.0]);
    // Widening the largest finite F32 rate must not retain a four-byte parameter ABI.
    // MAX_F32 * 2^-128 = 1-2^-24 exactly, so the updated weight is2^-24.
    let expected = f64::from_bits((1023 - 24) << 52);
    verify(
        &source::extreme_rate(&[1.0], &[f64::from_bits((1023 - 128) << 52)]).unwrap(),
        &[expected],
    );
    let exceptional = source::preserve(&[f64::INFINITY], &[f64::INFINITY]).unwrap();
    let mut observed = [0.0_f64];
    exceptional.read_into(&mut observed).unwrap();
    assert!(observed[0].is_infinite());
    verify(&source::preserve(&[1.0], &[1.0]).unwrap(), &[1.25]);
    for error in [
        source::checked(&[1.0], &[1.0]).unwrap_err(),
        source::strict_native(&[1.0], &[1.0]).unwrap_err(),
        source::portable(&[1.0], &[1.0]).unwrap_err(),
        source::tight(&[1.0], &[1.0]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"));
    }
    assert!(source::preserve::<f64, 0>(&[], &[]).is_err());
    let w = source::identity(&[1.0_f64, 2.0]).unwrap();
    let g = source::identity(&[1.0_f64, 2.0, 3.0]).unwrap();
    assert!(source::preserve::<f64, 2>(&w, &g).is_err());
    verify(&w, &[1.0, 2.0]);
    verify(&g, &[1.0, 2.0, 3.0]);
    global::clear_thread_cache().unwrap();
}

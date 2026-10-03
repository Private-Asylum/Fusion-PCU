//! Required read span is a prefix guarantee, distinct from retained backing extent.
use super::super::*;

#[crate::pcu(crate_path = crate, invocations = 4)]
fn indexed(q: &mut [i64], input: &[i64; 8], r: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[id]);
    r[id] = remainder;
    q[id] = quotient;
}

#[crate::pcu(crate_path = crate, invocations = 4)]
fn element_zero(q: &mut [i64], input: &[i64; 8], r: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[0], input[0]);
    q[id] = quotient;
    r[id] = remainder;
}

#[crate::pcu(crate_path = crate, invocations = 4)]
fn canonical(input: &[i64; 8], rhs: &[i64; 12], q: &mut [i64], r: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], rhs[id]);
    q[id] = quotient;
    r[id] = remainder;
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires native MLX runtime on Apple silicon"
)]
fn resident_array_extent_may_exceed_the_actual_read_span() {
    let _guard = global::policy::TEST_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let root = root();
    // These suffix values would fault if fetched as operands. They remain
    // ordinary retained input bytes, without participating in this operation.
    let encoded = [17_i64, 19, 21, 23, 0, i64::MIN, -1, 0];
    let input = owner(&root, &encoded);
    let mut q = owner(&root, &[91_i64; 6]);
    let mut r = owner(&root, &[92_i64; 9]);
    indexed(&mut q, &input, &mut r).unwrap();
    assert_eq!(values::<6>(&q), [1, 1, 1, 1, 91, 91]);
    assert_eq!(values::<9>(&r), [0, 0, 0, 0, 92, 92, 92, 92, 92]);
    let mut host_q = [93_i64; 7];
    element_zero(&mut host_q, &input, &mut r).unwrap();
    assert_eq!(host_q, [1, 1, 1, 1, 93, 93, 93]);
    assert_eq!(values::<9>(&r), [0, 0, 0, 0, 92, 92, 92, 92, 92]);
    assert_eq!(values::<8>(&input), encoded);

    // Distinct original full shapes also work through the canonical factory.
    // Changing owners at identical shapes reuses the cold specialization.
    let rhs_values = [3_i64, 5, 7, 11, 0, 0, 0, 0, 0, 0, 0, 0];
    let rhs = owner(&root, &rhs_values);
    for bank in 0..3_i64 {
        let lhs_values = [
            17 + bank,
            19 + bank,
            21 + bank,
            23 + bank,
            0,
            i64::MIN,
            -1,
            0,
        ];
        let lhs = owner(&root, &lhs_values);
        canonical(&lhs, &rhs, &mut q, &mut r).unwrap();
        let expected_q = core::array::from_fn::<_, 4, _>(|id| lhs_values[id] / rhs_values[id]);
        let expected_r = core::array::from_fn::<_, 4, _>(|id| lhs_values[id] % rhs_values[id]);
        assert_eq!(&values::<6>(&q)[..4], &expected_q);
        assert_eq!(&values::<9>(&r)[..4], &expected_r);
        assert_eq!(&values::<6>(&q)[4..], &[91, 91]);
        assert_eq!(&values::<9>(&r)[4..], &[92; 5]);
        assert_eq!(values::<8>(&lhs), lhs_values);
    }
    assert_eq!(values::<12>(&rhs), rhs_values);

    let before_q = values::<6>(&q);
    let before_r = values::<9>(&r);
    let bad = owner(&root, &[0_i64, 19, 21, 23, 0, i64::MIN, -1, 0]);
    assert!(matches!(
        element_zero(&mut q, &bad, &mut r),
        Err(global::PcuExecutionError::ArithmeticFault(fault))
            if fault.kind == crate::PcuExecutionFaultKind::DivideByZero
                && fault.invocation_id == 0 && !fault.recovered
    ));
    assert_eq!(values::<6>(&q), before_q);
    assert_eq!(values::<9>(&r), before_r);
    element_zero(&mut q, &input, &mut r).unwrap();
    assert_eq!(&values::<6>(&q)[..4], &[1; 4]);
    assert_eq!(&values::<9>(&r)[..4], &[0; 4]);
    global::clear_thread_cache().unwrap();
    drop((input, rhs, bad, q, r));
    global::use_defaults().unwrap();
}

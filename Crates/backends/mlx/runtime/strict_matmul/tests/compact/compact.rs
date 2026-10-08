//! Bounded larger-inner independent ordered oracle and actual receipt-memory controls.
use super::*;

fn reference<T: PcuCheckedFloat>(
    left: &[T],
    right: &[T],
    inner: usize,
    zero: T,
    policy: PcuFloatUnderflowPolicy,
) -> Result<Vec<T>, fusion_pcu::PcuExecutionFault> {
    let mut output = Vec::new();
    for cell in 0..4 {
        let mut accumulator = zero;
        for reduction in 0..inner {
            let ordinal = u64::try_from((cell * inner + reduction) * 2).unwrap();
            let fault = |kind, invocation_id| fusion_pcu::PcuExecutionFault {
                kind,
                invocation_id,
                recovered: false,
            };
            let product = left[cell / 2 * inner + reduction]
                .pcu_checked_mul_with_policy(right[reduction * 2 + cell % 2], policy)
                .map_err(|kind| fault(kind, ordinal))?;
            accumulator = accumulator
                .pcu_checked_add_with_policy(product, policy)
                .map_err(|kind| fault(kind, ordinal + 1))?;
        }
        output.push(accumulator);
    }
    Ok(output)
}
fn banks<T: PcuCheckedFloat>(inner: usize, values: [T; 7]) -> Vec<(Vec<T>, Vec<T>)> {
    let [zero, one, two, half, max, tiny, invalid] = values;
    let mut cases = vec![(vec![zero; 2 * inner], vec![one; 2 * inner])];
    let mut left = vec![zero; 2 * inner];
    let mut right = vec![zero; 2 * inner];
    left[2 * inner - 1] = max;
    right[2 * inner - 1] = two;
    cases.push((left, right));
    let mut left = vec![zero; 2 * inner];
    let mut right = vec![zero; 2 * inner];
    left[2 * inner - 2] = max;
    left[2 * inner - 1] = max;
    right[2 * inner - 3] = one;
    right[2 * inner - 1] = one;
    cases.push((left, right));
    let mut left = vec![zero; 2 * inner];
    let mut right = vec![zero; 2 * inner];
    left[2 * inner - 1] = tiny;
    right[2 * inner - 1] = half;
    cases.push((left, right));
    let mut left = vec![zero; 2 * inner];
    let mut right = vec![zero; 2 * inner];
    left[2 * inner - 1] = tiny;
    right[2 * inner - 1] = one;
    cases.push((left, right));
    let mut left = vec![zero; 2 * inner];
    let mut right = vec![zero; 2 * inner];
    left[inner] = invalid;
    right[2 * inner - 1] = invalid;
    cases.push((left, right));
    cases
}
#[allow(clippy::too_many_lines)] // One native extent/policy matrix retains independently derived ordered faults and escaped owners.
fn matrix<T: PcuCheckedFloat>(session: &MlxSession, values: [T; 7]) {
    let zero = values[0];
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for inner in [17, 31, 128, 1024, 65535] {
            let map = session
                .prepare_strict_matmul(T::TYPE, policy, [2, inner, 2])
                .unwrap();
            assert_eq!(
                map.fault_domain().event_extent(),
                u64::try_from(8 * inner).unwrap()
            );
            assert_eq!(map.status_word_extent(), 8);
            eprintln!(
                "mlx strict MatMul status-memory: inner={inner}, semantic_event_words={}, native_receipt_words={}, native_status_bytes={}",
                map.fault_domain().event_extent(),
                map.status_word_extent(),
                map.status_word_extent() * 4
            );
            let cases = banks(inner, values);
            let good_left = session.upload_encoded(&cases[0].0).unwrap();
            let good_right = session.upload_encoded(&cases[0].1).unwrap();
            let retained = map.execute_resident(&good_left, &good_right).unwrap();
            for (left, right) in cases {
                let a = session.upload_encoded(&left).unwrap();
                let b = session.upload_encoded(&right).unwrap();
                let actual = map.execute_resident(&a, &b);
                match reference(&left, &right, inner, zero, policy) {
                    Ok(expected) => {
                        let owner = actual.unwrap();
                        let mut host = [values[1]; 6];
                        owner.output().read_into(&mut host).unwrap();
                        assert_eq!(bytes(&host[..4]), bytes(&expected));
                        assert_eq!(bytes(&host[4..]), bytes(&[values[1]; 2]));
                    }
                    Err(expected) => assert!(
                        matches!(actual,Err(MlxError::Arithmetic(fault)) if fault==expected)
                    ),
                }
                let mut old = [values[1]; 6];
                retained.output().read_into(&mut old).unwrap();
                assert_eq!(bytes(&old[..4]), bytes(&[zero; 4]));
                assert_eq!(bytes(&old[4..]), bytes(&[values[1]; 2]));
                let retry = map.execute_resident(&good_left, &good_right).unwrap();
                let mut host = [values[1]; 6];
                retry.output().read_into(&mut host).unwrap();
                assert_eq!(bytes(&host[..4]), bytes(&[zero; 4]));
                assert_eq!(bytes(&host[4..]), bytes(&[values[1]; 2]));
            }
            drop(map);
            let mut old = [values[1]; 6];
            retained.output().read_into(&mut old).unwrap();
            assert_eq!(bytes(&old[..4]), bytes(&[zero; 4]));
            assert_eq!(bytes(&old[4..]), bytes(&[values[1]; 2]));
        }
    }
}
#[test]
#[ignore = "Required actual bounded larger-inner compact MatMul and original late step faults, status memory and retained owners."]
fn compact_matmul_native_large_inner_ordered_faults_and_status_memory() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    matrix(
        &session,
        [
            0.0_f32,
            1.0,
            2.0,
            0.5,
            f32::MAX,
            f32::from_bits(1),
            f32::NAN,
        ],
    );
    matrix(
        &session,
        [
            0.0_f64,
            1.0,
            2.0,
            0.5,
            f64::MAX,
            f64::from_bits(1),
            f64::NAN,
        ],
    );
}

fn malformed<T: PcuCheckedFloat>(session: &MlxSession, zero: T, one: T) {
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        let good = session
            .prepare_strict_matmul(T::TYPE, policy, [1, 3, 2])
            .unwrap();
        let left = session.upload_encoded(&[zero; 3]).unwrap();
        let right = session.upload_encoded(&[zero; 6]).unwrap();
        let old = good.execute_resident(&left, &right).unwrap();
        for words in [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [1, 1, 0, 0],
            [3, 4, 0, 0],
            [0, 0x101, 0, 0],
            [6, 1, 0, 0],
            [0, 0, 5, 1],
            [0, 0, 12, 1],
            [0, 1, 6, u32::MAX],
        ] {
            let bad = crate::ffi::StrictMatMul::prepare_receipt_fixture(
                &session.0.native,
                T::TYPE,
                policy,
                [1, 3, 2],
                &words,
            )
            .unwrap();
            assert!(matches!(
                bad.execute(left.native(), right.native()),
                Err(MlxError::Abi(_))
            ));
            let retry = good.execute_resident(&left, &right).unwrap();
            let mut actual = [one; 4];
            retry.output().read_into(&mut actual).unwrap();
            assert_eq!(bytes(&actual[..2]), bytes(&[zero; 2]));
            assert_eq!(bytes(&actual[2..]), bytes(&[one; 2]));
            let mut actual = [one; 4];
            old.output().read_into(&mut actual).unwrap();
            assert_eq!(bytes(&actual[..2]), bytes(&[zero; 2]));
            assert_eq!(bytes(&actual[2..]), bytes(&[one; 2]));
        }
    }
}
#[test]
#[ignore = "Required actual malformed compact cell receipts, checked private cleanup, healthy retry and retained output."]
fn compact_matmul_native_malformed_receipts_and_healthy_retry() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    malformed(&session, 0.0_f32, 1.0_f32);
    malformed(&session, 0.0_f64, 1.0_f64);
}

#[test]
fn compact_matmul_reference_covers_original_late_steps_and_cell_order() {
    fn check<T: PcuCheckedFloat>(values: [T; 7]) {
        for inner in [17, 31, 128, 1024, 65535] {
            let cases = banks(inner, values);
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                assert_eq!(
                    bytes(&reference(&cases[0].0, &cases[0].1, inner, values[0], policy).unwrap()),
                    bytes(&[values[0]; 4])
                );
                for (case, kind, ordinal) in [
                    (1, PcuExecutionFaultKind::ArithmeticOverflow, 8 * inner - 2),
                    (2, PcuExecutionFaultKind::ArithmeticOverflow, 8 * inner - 1),
                    (
                        5,
                        PcuExecutionFaultKind::InvalidFloatingOperand,
                        4 * inner - 2,
                    ),
                ] {
                    let fault = reference(&cases[case].0, &cases[case].1, inner, values[0], policy)
                        .err()
                        .unwrap();
                    assert_eq!(fault.kind, kind);
                    assert_eq!(fault.invocation_id, u64::try_from(ordinal).unwrap());
                    assert!(!fault.recovered);
                }
                for case in [3, 4] {
                    let actual =
                        reference(&cases[case].0, &cases[case].1, inner, values[0], policy);
                    if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                        || case == 3 && policy == PcuFloatUnderflowPolicy::IeeeAfterRounding
                    {
                        let fault = actual.err().unwrap();
                        assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
                        assert_eq!(fault.invocation_id, u64::try_from(8 * inner - 2).unwrap());
                    } else {
                        let last = if case == 4 { values[5] } else { values[0] };
                        assert_eq!(
                            bytes(&actual.unwrap()),
                            bytes(&[values[0], values[0], values[0], last])
                        );
                    }
                }
            }
        }
    }
    check([
        0.0_f32,
        1.0,
        2.0,
        0.5,
        f32::MAX,
        f32::from_bits(1),
        f32::NAN,
    ]);
    check([
        0.0_f64,
        1.0,
        2.0,
        0.5,
        f64::MAX,
        f64::from_bits(1),
        f64::NAN,
    ]);
}

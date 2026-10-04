use super::*;
use fusion_pcu::{PcuCheckedFloat, PcuCheckedFloatWidening, PcuExecutionFault, PcuExecutionFaultKind};
fn bytes<T: PcuCheckedFloat>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn oracle<T: PcuCheckedFloat>(
    weight: T,
    gradient: T,
    rate: T,
    policy: PcuFloatUnderflowPolicy,
    element: usize,
) -> Result<T, PcuExecutionFault> {
    let ordinal = u64::try_from(element * 2).unwrap();
    let fault = |kind, invocation_id| PcuExecutionFault {
        kind,
        invocation_id,
        recovered: false,
    };
    let product = gradient
        .pcu_checked_mul_with_policy(rate, policy)
        .map_err(|kind| fault(kind, ordinal))?;
    weight
        .pcu_checked_sub_with_policy(product, policy)
        .map_err(|kind| fault(kind, ordinal + 1))
}

#[test]
fn strict_sgd_event_domain_rejects_impossible_status_recovered_and_wrong_extent() {
    let domain = TensorStrictFaultDomain::sgd(
        PcuScalarType::F64,
        7,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    )
    .unwrap();
    assert!(validate(&[0; 14], domain).is_ok());
    assert!(validate(&[0; 13], domain).is_err());
    for record in [2, 0x101, u32::MAX] {
        let mut words = [0; 14];
        words[2] = record;
        assert!(matches!(
            validate(&words, domain),
            Err(MetalError::Runtime(_))
        ));
    }
    let mut words = [0; 14];
    words[3] = 4;
    words[4] = 3;
    assert!(
        matches!(validate(&words,domain),Err(MetalError::Arithmetic(fault)) if fault.invocation_id==3)
    );
}
fn tag(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 1,
        PcuExecutionFaultKind::ArithmeticUnderflow => 3,
        PcuExecutionFaultKind::InvalidFloatingOperand => 4,
        _ => panic!("SGD step fault"),
    }
}
fn verify<T: PcuCheckedFloat>(
    map: &MetalPreparedStrictSgd,
    weights: &[T; 7],
    gradients: &[T; 7],
    rate: T,
    policy: PcuFloatUnderflowPolicy,
) {
    let session = &map.session;
    let a_bytes = bytes(weights);
    let b_bytes = bytes(gradients);
    let a = session.upload_bytes(&a_bytes).unwrap();
    let b = session.upload_bytes(&b_bytes).unwrap();
    let output = session.allocate_zeroed_bytes(map.output_bytes).unwrap();
    let records = session.0.native.allocate(14 * 4).unwrap();
    records.fill_ones();
    session
        .0
        .native
        .execute(
            &map.pipeline,
            [&a.native, &b.native, &output.native, &records],
            [7, map.policy],
            7,
        )
        .unwrap();
    let actual = records.read(14).unwrap();
    let mut payload = vec![0; map.output_bytes];
    output.read_into_bytes(&mut payload).unwrap();
    let width = usize::from(T::TYPE.bit_width()) / 8;
    let mut expected = [0; 14];
    for index in 0..7 {
        match oracle(weights[index], gradients[index], rate, policy, index) {
            Ok(value) => assert_eq!(
                &payload[index * width..(index + 1) * width],
                value.encode_le().as_ref()
            ),
            Err(fault) => expected[usize::try_from(fault.invocation_id).unwrap()] = tag(fault.kind),
        }
    }
    assert_eq!(actual, expected);
    match map.execute_completed(&a, &b) {
        Ok(owner) => {
            assert!(expected.iter().all(|&word| word == 0));
            let mut bytes = vec![0; map.output_bytes];
            owner.read_into_bytes(&mut bytes).unwrap();
            assert_eq!(bytes, payload);
        }
        Err(MetalError::Arithmetic(fault)) => {
            assert_eq!(
                fault.invocation_id,
                u64::try_from(expected.iter().position(|&word| word != 0).unwrap()).unwrap()
            );
            assert!(map.domain.accepts(fault));
        }
        Err(error) => panic!("unexpected {error:?}"),
    }
    let mut original = vec![0; a_bytes.len()];
    a.read_into_bytes(&mut original).unwrap();
    assert_eq!(original, a_bytes);
    b.read_into_bytes(&mut original).unwrap();
    assert_eq!(original, b_bytes);
}
fn native<T: PcuCheckedFloat>(
    session: &MetalSession,
    foreign: &MetalSession,
    edges: &[T],
    from_bits: impl Fn(u64) -> T,
    from_rate: impl Fn(f32) -> T,
) {
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for rate in [0f32, -0f32, 0.5, 1., -1., f32::MAX, f32::from_bits(1)] {
            let map = session
                .prepare_strict_sgd(T::TYPE, 7, rate, policy)
                .unwrap();
            for &a in edges {
                for &b in edges {
                    verify(&map, &[a; 7], &[b; 7], from_rate(rate), policy);
                }
            }
            let mut seed = 0x5893_17bd_c133_229au64;
            for _ in 0..128 {
                let mut next = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    from_bits(seed)
                };
                let weights = core::array::from_fn(|_| next());
                let gradients = core::array::from_fn(|_| next());
                verify(&map, &weights, &gradients, from_rate(rate), policy);
            }
            let a = session.upload_bytes(&bytes(&[edges[0]; 7])).unwrap();
            let b = foreign.upload_bytes(&bytes(&[edges[0]; 7])).unwrap();
            assert!(matches!(
                map.execute_completed(&a, &b),
                Err(MetalError::ForeignSession)
            ));
            let short = session.allocate_zeroed_bytes(1).unwrap();
            assert!(matches!(
                map.execute_completed(&short, &a),
                Err(MetalError::InvalidExtent)
            ));
        }
    }
}
#[test]
#[ignore = "Requires native Metal ordered SGD encoded steps and private terminal ownership."]
fn strict_sgd_native_f32_f64_all_policies_rates_step_oracle_and_old_owners() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    let bits32 = [
        0,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        1,
        0x0080_0000,
        0x3f00_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7fc0_1234,
        0x4040_0000,
    ];
    native(
        &session,
        &foreign,
        &bits32.map(f32::from_bits),
        |bits| f32::from_bits(u32::try_from(bits & 0xffff_ffff).unwrap()),
        |rate| rate,
    );
    let bits64 = [
        0,
        0x8000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0xbff0_0000_0000_0000,
        1,
        0x0010_0000_0000_0000,
        0x3fe0_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0x7ff8_0000_0000_1234,
        0x4008_0000_0000_0000,
    ];
    native(
        &session,
        &foreign,
        &bits64.map(f64::from_bits),
        f64::from_bits,
        |rate| rate.pcu_checked_to_f64().unwrap(),
    );
    for rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            session
                .prepare_strict_sgd(
                    PcuScalarType::F64,
                    7,
                    rate,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding
                )
                .is_err()
        );
    }
    assert!(
        session
            .prepare_strict_sgd(
                PcuScalarType::F64,
                0,
                1.,
                PcuFloatUnderflowPolicy::IeeeAfterRounding
            )
            .is_err()
    );
}

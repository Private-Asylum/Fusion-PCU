use super::*;
use fusion_pcu::{PcuCheckedFloat, PcuCheckedFloatWidening, PcuExecutionFault};
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

use crate::MlxRuntime;
use fusion_pcu::dialect::tensor::TensorStrictFaultDomain;
#[test]
fn strict_sgd_event_domain_rejects_impossible_status_recovered_and_wrong_extent() {
    let domain = TensorStrictFaultDomain::sgd(
        PcuScalarType::F64,
        7,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    )
    .unwrap();
    assert!(crate::ffi::StrictSgd::validate_domain(&[0; 14], domain).is_ok());
    assert!(crate::ffi::StrictSgd::validate_domain(&[0; 13], domain).is_err());
    for record in [2, 0x101, u32::MAX] {
        let mut words = [0; 14];
        words[2] = record;
        assert!(crate::ffi::StrictSgd::validate_domain(&words, domain).is_err());
    }
}
fn verify<T: PcuCheckedFloat>(
    session: &MlxSession,
    map: &MlxPreparedStrictSgd,
    weights: &[T; 7],
    gradients: &[T; 7],
    rate: T,
    policy: PcuFloatUnderflowPolicy,
) {
    let a = session.upload_encoded(weights).unwrap();
    let b = session.upload_encoded(gradients).unwrap();
    let expected = (0..7)
        .map(|index| oracle(weights[index], gradients[index], rate, policy, index))
        .collect::<Result<Vec<_>, _>>();
    match expected {
        Ok(expected) => {
            let result = map.execute_resident(&a, &b).unwrap();
            assert!(result.recovered_fault().is_none());
            let mut actual = vec![weights[0]; 9];
            result.output().read_into(&mut actual).unwrap();
            assert_eq!(bytes(&actual[..7]), bytes(&expected));
            assert_eq!(bytes(&actual[7..]), bytes(&[weights[0]; 2]));
        }
        Err(expected) => assert!(
            matches!(map.execute_resident(&a,&b),Err(MlxError::Arithmetic(fault)) if fault==expected)
        ),
    }
    let mut original = *weights;
    a.read_into(&mut original).unwrap();
    assert_eq!(bytes(&original), bytes(weights));
    b.read_into(&mut original).unwrap();
    assert_eq!(bytes(&original), bytes(gradients));
}
fn native<T: PcuCheckedFloat>(
    session: &MlxSession,
    foreign: &MlxSession,
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
                .prepare_strict_sgd(T::TYPE, policy, 7, rate)
                .unwrap();
            for &a in edges {
                for &b in edges {
                    verify(session, &map, &[a; 7], &[b; 7], from_rate(rate), policy);
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
                verify(session, &map, &weights, &gradients, from_rate(rate), policy);
            }
            let a = session.upload_encoded(&[edges[0]; 7]).unwrap();
            let b = foreign.upload_encoded(&[edges[0]; 7]).unwrap();
            assert!(matches!(
                map.execute_resident(&a, &b),
                Err(MlxError::ForeignSession)
            ));
            let short = session.upload_encoded(&[edges[0]; 1]).unwrap();
            assert!(matches!(
                map.execute_resident(&short, &a),
                Err(MlxError::InvalidExtent)
            ));
        }
    }
}
#[test]
#[ignore = "Requires pinned native MLX GPU ordered SGD step oracle and immutable owners."]
fn strict_sgd_native_f32_f64_all_policies_rates_step_oracle_and_old_owners() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
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
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    7,
                    rate
                )
                .is_err()
        );
    }
    assert!(
        session
            .prepare_strict_sgd(
                PcuScalarType::F64,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                0,
                1.
            )
            .is_err()
    );
}

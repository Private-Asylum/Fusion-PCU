#[test]
fn strict_mse_event_domain_refuses_invalid_records_and_preserves_order() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            for count in 1..=10 {
                let domain = TensorStrictFaultDomain::mse(scalar, count, policy).unwrap();
                let events = usize::try_from(domain.event_extent()).unwrap();
                assert_eq!(events, usize::try_from(count * 3 + 1).unwrap());
                let mut records = vec![0; events];
                assert!(super::validate(&records, domain).is_ok());
                for ordinal in 0..events {
                    records[ordinal] = 0x100;
                    assert!(super::validate(&records, domain).is_err());
                    records[ordinal] = 0;
                    let location = domain.location(u64::try_from(ordinal).unwrap()).unwrap();
                    assert_eq!(
                        domain.ordinal(location),
                        Some(u64::try_from(ordinal).unwrap())
                    );
                }
                assert!(super::validate(&records[..events - 1], domain).is_err());
            }
        }
    }
    assert!(
        TensorStrictFaultDomain::mse(
            PcuScalarType::F32,
            0,
            PcuFloatUnderflowPolicy::IeeeAfterRounding
        )
        .is_none()
    );
}

use super::*;
use fusion_pcu::{PcuCheckedFloat, PcuExecutionFault, PcuExecutionFaultKind};
fn bytes<T: PcuCheckedFloat>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn oracle<T: PcuCheckedFloat>(
    left: &[T],
    right: &[T],
    zero: T,
    count: T,
    policy: PcuFloatUnderflowPolicy,
) -> Result<T, PcuExecutionFault> {
    let fault = |kind, invocation_id| PcuExecutionFault {
        kind,
        invocation_id,
        recovered: false,
    };
    let mut sum = zero;
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        let ordinal = u64::try_from(index * 3).unwrap();
        let difference = left
            .pcu_checked_sub_with_policy(right, policy)
            .map_err(|kind| fault(kind, ordinal))?;
        let squared = difference
            .pcu_checked_mul_with_policy(difference, policy)
            .map_err(|kind| fault(kind, ordinal + 1))?;
        sum = sum
            .pcu_checked_add_with_policy(squared, policy)
            .map_err(|kind| fault(kind, ordinal + 2))?;
    }
    sum.pcu_checked_div_with_policy(count, policy)
        .map_err(|kind| fault(kind, u64::try_from(left.len() * 3).unwrap()))
}

fn tag(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 1,
        PcuExecutionFaultKind::ArithmeticUnderflow => 3,
        PcuExecutionFaultKind::InvalidFloatingOperand => 4,
        _ => panic!("MSE fault"),
    }
}
fn verify<T: PcuCheckedFloat>(
    map: &MetalPreparedStrictMse,
    left: &[T],
    right: &[T],
    zero: T,
    count: T,
    policy: PcuFloatUnderflowPolicy,
) {
    let session = &map.session;
    let a_bytes = bytes(left);
    let b_bytes = bytes(right);
    let a = session.upload_bytes(&a_bytes).unwrap();
    let b = session.upload_bytes(&b_bytes).unwrap();
    let output = session.allocate_zeroed_bytes(map.output_bytes).unwrap();
    let records = session.0.native.allocate(8).unwrap();
    records.fill_ones();
    session
        .0
        .native
        .execute(
            &map.pipeline,
            [&a.native, &b.native, &output.native, &records],
            [1, map.policy],
            1,
        )
        .unwrap();
    let actual = records.read(2).unwrap();
    let expected = oracle(left, right, zero, count, policy);
    let expected_records = if let Err(fault) = expected {
        [u32::try_from(fault.invocation_id).unwrap(), tag(fault.kind)]
    } else {
        [0; 2]
    };
    assert_eq!(actual, expected_records);
    match expected {
        Ok(expected) => {
            let owner = map.execute_completed(&a, &b).unwrap();
            let mut output = vec![0; map.output_bytes];
            owner.read_into_bytes(&mut output).unwrap();
            assert_eq!(&output[..map.output_bytes], expected.encode_le().as_ref());
        }
        Err(expected) => assert!(
            matches!(map.execute_completed(&a,&b),Err(MetalError::Arithmetic(fault)) if fault==expected)
        ),
    }
    let mut old = vec![0; a_bytes.len()];
    a.read_into_bytes(&mut old).unwrap();
    assert_eq!(old, a_bytes);
    b.read_into_bytes(&mut old).unwrap();
    assert_eq!(old, b_bytes);
}

fn native<T: PcuCheckedFloat>(
    session: &MetalSession,
    foreign: &MetalSession,
    edges: &[T],
    from_bits: impl Fn(u64) -> T,
    from_count: impl Fn(u8) -> T,
) {
    let zero = edges[0];
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for count in [1usize, 7, 10] {
            let map = session.prepare_strict_mse(T::TYPE, count, policy).unwrap();
            let denominator = from_count(u8::try_from(count).unwrap());
            for &a in edges {
                for &b in edges {
                    verify(
                        &map,
                        &vec![a; count],
                        &vec![b; count],
                        zero,
                        denominator,
                        policy,
                    );
                }
            }
            let mut seed = 0x18a9_bc35_12de_507fu64;
            for _ in 0..128 {
                let mut next = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    from_bits(seed)
                };
                let a = (0..count).map(|_| next()).collect::<Vec<_>>();
                let b = (0..count).map(|_| next()).collect::<Vec<_>>();
                verify(&map, &a, &b, zero, denominator, policy);
            }
            let a = session.upload_bytes(&bytes(&vec![zero; count])).unwrap();
            let b = foreign.upload_bytes(&bytes(&vec![zero; count])).unwrap();
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
    for count in [0, 65536] {
        assert!(
            session
                .prepare_strict_mse(T::TYPE, count, PcuFloatUnderflowPolicy::IeeeAfterRounding)
                .is_err()
        );
    }
}
#[test]
#[ignore = "Requires actual native encoded ordered MSE reduction, mean division and private owners."]
fn strict_mse_native_f32_f64_ordered_reduction_mean_oracle_and_old_owners() {
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
        f32::from,
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
        f64::from,
    );
}

#[test]
fn compact_mse_receipt_domain_and_canonical_success() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            for count in [1, 11, 65535] {
                let domain = TensorStrictFaultDomain::mse(scalar, count, policy).unwrap();
                let decode = |words: &[u32]| super::validate_receipt(words, domain);
                assert!(decode(&[0, 0]).is_ok());
                assert!(
                    matches!(decode(&[0,4]),Err(MetalError::Arithmetic(f)) if f.invocation_id==0&&!f.recovered)
                );
                let last = u32::try_from(domain.event_extent() - 1).unwrap();
                for words in [
                    [1, 0],
                    [0, 0x104],
                    [0, 2],
                    [u32::MAX, 4],
                    [last + 1, 4],
                    [last, 1],
                    [2, 3],
                    [1, 4],
                ] {
                    assert!(decode(&words).is_err());
                }
                if count == 1 {
                    assert!(decode(&[last, 3]).is_err());
                }
                assert!(decode(&[0]).is_err());
                assert!(decode(&[0, 0, 0]).is_err());
            }
        }
    }
}

#[allow(clippy::too_many_arguments)] // Independent scalar edge controls are explicit test oracle parameters.
fn large<T: PcuCheckedFloat>(
    session: &MetalSession,
    foreign: &MetalSession,
    zero: T,
    half: T,
    nan: T,
    max: T,
    negative_max: T,
    big: T,
    tiny_root: T,
    from_count: impl Fn(u16) -> T,
) {
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for invalid in [0, 65536] {
            assert!(matches!(
                session.prepare_strict_mse(T::TYPE, invalid, policy),
                Err(MetalError::InvalidExtent)
            ));
        }
        for count in [11usize, 16, 31, 64, 255, 256, 1024, 65535] {
            let map = session.prepare_strict_mse(T::TYPE, count, policy).unwrap();
            let denominator = from_count(u16::try_from(count).unwrap());
            let right = vec![zero; count];
            let input = session.upload_bytes(&bytes(&right)).unwrap();
            let alien = foreign.upload_bytes(&bytes(&right)).unwrap();
            assert!(matches!(
                map.execute_completed(&input, &alien),
                Err(MetalError::ForeignSession)
            ));
            let short = session
                .allocate_zeroed_bytes((count - 1) * usize::from(T::TYPE.bit_width()) / 8)
                .unwrap();
            assert!(matches!(
                map.execute_completed(&short, &input),
                Err(MetalError::InvalidExtent)
            ));
            let retained = map.execute_completed(&input, &input).unwrap();
            let mut left = vec![zero; count];
            verify(&map, &left, &right, zero, denominator, policy);
            left.fill(half);
            verify(&map, &left, &right, zero, denominator, policy);
            left.fill(zero);
            left[count - 1] = nan;
            verify(&map, &left, &right, zero, denominator, policy);
            left.fill(zero);
            left[count - 1] = max;
            let mut opposite = right.clone();
            opposite[count - 1] = negative_max;
            verify(&map, &left, &opposite, zero, denominator, policy);
            left.fill(zero);
            left[count - 2] = max;
            left[count - 1] = nan;
            verify(&map, &left, &right, zero, denominator, policy);
            left.fill(zero);
            left[count - 3] = big;
            left[count - 2] = big;
            left[count - 1] = nan;
            verify(&map, &left, &right, zero, denominator, policy);
            left.fill(zero);
            left[0] = nan;
            left[count - 1] = max;
            verify(&map, &left, &right, zero, denominator, policy);
            left.fill(zero);
            left[0] = tiny_root;
            verify(&map, &left, &right, zero, denominator, policy);
            left.fill(half);
            verify(&map, &left, &right, zero, denominator, policy);
            let mut actual = vec![0xff; map.output_bytes + 3];
            assert!(matches!(
                retained.read_into_bytes(&mut actual),
                Err(MetalError::InvalidExtent)
            ));
            assert!(actual.iter().all(|b| *b == 0xff));
            retained
                .read_into_bytes(&mut actual[..map.output_bytes])
                .unwrap();
            assert!(actual[..map.output_bytes].iter().all(|b| *b == 0));
            assert_eq!(&actual[map.output_bytes..], &[0xff; 3]);
        }
    }
}
#[test]
#[ignore = "Requires actual Apple GPU compact MSE, up to65535 cells, independent ordered scalar oracle."]
fn strict_mse_compact_large_native_order_mean_and_retry() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    large(
        &session,
        &foreign,
        0.0_f32,
        0.5,
        f32::from_bits(0x7fc0_0001),
        f32::MAX,
        -f32::MAX,
        f32::from_bits(0x5f40_0000),
        f32::from_bits(0x1a80_0000),
        f32::from,
    );
    large(
        &session,
        &foreign,
        0.0_f64,
        0.5,
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::MAX,
        -f64::MAX,
        f64::from_bits(0x5fe8_0000_0000_0000),
        f64::from_bits(0x1e60_0000_0000_0000),
        f64::from,
    );
}

#[test]
#[ignore = "Requires actual GPU private compact receipt malformed-domain/refusal and healthy retry."]
fn strict_mse_compact_native_malformed_receipts() {
    let session = MetalSession::open(0).unwrap();
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for count in [1usize, 11, 65535] {
            let policy = PcuFloatUnderflowPolicy::IeeeAfterRounding;
            let domain =
                TensorStrictFaultDomain::mse(scalar, u64::try_from(count).unwrap(), policy)
                    .unwrap();
            let last = u32::try_from(domain.event_extent() - 1).unwrap();
            let mut receipts = vec![
                [1, 0],
                [0, 0x104],
                [0, 2],
                [u32::MAX, 4],
                [last + 1, 4],
                [last, 1],
                [2, 3],
                [1, 4],
            ];
            if count == 1 {
                receipts.push([last, 3]);
            }
            for receipt in receipts {
                let mut map = session.prepare_strict_mse(scalar, count, policy).unwrap();
                let code = format!(
                    "#include <metal_stdlib>\nusing namespace metal;kernel void malformed(device const uchar* a [[buffer(0)]],device const uchar* b [[buffer(1)]],device uint* output [[buffer(2)]],device uint* records [[buffer(3)]],constant uint2& p [[buffer(4)]],uint id [[thread_position_in_grid]]){{if(id!=0u)return;records[0]={}u;records[1]={}u;output[0]=0u;{}}}",
                    receipt[0],
                    receipt[1],
                    if scalar == PcuScalarType::F64 {
                        "output[1]=0u;"
                    } else {
                        ""
                    }
                );
                map.pipeline = session.0.native.compile(&code, "malformed").unwrap();
                let input = session.allocate_zeroed_bytes(map.input_bytes[0]).unwrap();
                assert!(matches!(
                    map.execute_completed(&input, &input),
                    Err(MetalError::Runtime(_))
                ));
                let healthy = session.prepare_strict_mse(scalar, count, policy).unwrap();
                let output = healthy.execute_completed(&input, &input).unwrap();
                let mut bytes = vec![0xff; map.output_bytes + 3];
                assert!(matches!(
                    output.read_into_bytes(&mut bytes),
                    Err(MetalError::InvalidExtent)
                ));
                assert!(bytes.iter().all(|b| *b == 0xff));
                output
                    .read_into_bytes(&mut bytes[..map.output_bytes])
                    .unwrap();
                assert!(bytes[..map.output_bytes].iter().all(|b| *b == 0));
                assert_eq!(&bytes[map.output_bytes..], &[0xff; 3]);
            }
        }
    }
}

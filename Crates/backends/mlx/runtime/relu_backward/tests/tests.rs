//! Actual MLX checked finite selection against independent core scalar law.
use super::*;
use fusion_pcu::PcuCheckedFloat;
fn bytes<T: PcuCheckedFloat>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn verify<T: PcuCheckedFloat>(
    session: &MlxSession,
    foreign: &MlxSession,
    left: &[T],
    right: &[T],
    policy: PcuFloatUnderflowPolicy,
    broadcast: [bool; 2],
) {
    let count = 5;
    let map = session
        .prepare_relu_backward(T::TYPE, policy, count, broadcast)
        .unwrap();
    let sizes = broadcast.map(|flag| if flag { 1 } else { count });
    let wrong = foreign.upload_encoded(&vec![left[0]; sizes[0]]).unwrap();
    let healthy = session.upload_encoded(&vec![right[0]; sizes[1]]).unwrap();
    assert!(matches!(
        map.execute_resident(&wrong, &healthy),
        Err(MlxError::ForeignSession)
    ));
    let oversized = session
        .upload_encoded(&vec![left[0]; sizes[0] + 1])
        .unwrap();
    assert!(matches!(
        map.execute_resident(&oversized, &healthy),
        Err(MlxError::InvalidExtent)
    ));
    for (&a, &b) in left.iter().zip(right) {
        let input = session.upload_encoded(&vec![a; sizes[0]]).unwrap();
        let upstream = session.upload_encoded(&vec![b; sizes[1]]).unwrap();
        let result = map.execute_resident(&input, &upstream);
        match a.pcu_checked_relu_backward_with_policy(b, policy) {
            Ok(expected) => {
                let completed = result.unwrap();
                assert!(completed.recovered_fault().is_none());
                assert_eq!(completed.output().scalar_type(), T::TYPE);
                let mut output = vec![a; count + 2];
                completed.output().read_into(&mut output).unwrap();
                assert_eq!(bytes(&output[..count]), bytes(&vec![expected; count]));
                assert_eq!(bytes(&output[count..]), bytes(&[a; 2]));
            }
            Err(kind) => assert!(
                matches!(result,Err(MlxError::Arithmetic(fault)) if fault.kind==kind&&!fault.recovered&&fault.invocation_id==0)
            ),
        }
        let mut original = vec![a; sizes[0]];
        input.read_into(&mut original).unwrap();
        assert_eq!(bytes(&original), bytes(&vec![a; sizes[0]]));
        let mut original = vec![b; sizes[1]];
        upstream.read_into(&mut original).unwrap();
        assert_eq!(bytes(&original), bytes(&vec![b; sizes[1]]));
    }
}
#[test]
#[ignore = "Requires actual MLX0.32.3 GPU backward-selection oracle."]
fn mlx_relu_backward_all_policies_broadcast_finite_masked_oracle_and_owner_guards() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let edges64 = [
        1.0_f64.to_bits(),
        0,
        0x8000_0000_0000_0000,
        1,
        0x8000_0000_0000_0001,
        0x0010_0000_0000_0000,
        (-1.0_f64).to_bits(),
        f64::MAX.to_bits(),
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        0x7ff8_0000_0000_0001,
    ];
    let edges32 = [
        1.0_f32.to_bits(),
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x0080_0000,
        (-1.0_f32).to_bits(),
        f32::MAX.to_bits(),
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
        0x7fc0_0001,
    ];
    let mut left64 = Vec::new();
    let mut right64 = Vec::new();
    let mut left32 = Vec::new();
    let mut right32 = Vec::new();
    for a in edges64 {
        for b in edges64 {
            left64.push(f64::from_bits(a));
            right64.push(f64::from_bits(b));
        }
    }
    for a in edges32 {
        for b in edges32 {
            left32.push(f32::from_bits(a));
            right32.push(f32::from_bits(b));
        }
    }
    let mut state = 0x1283_596a_d773_bb0f_u64;
    for _ in 0..64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        left64.push(f64::from_bits(state));
        left32.push(f32::from_bits(u32::try_from(state & 0xffff_ffff).unwrap()));
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        right64.push(f64::from_bits(state));
        right32.push(f32::from_bits(u32::try_from(state & 0xffff_ffff).unwrap()));
    }
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for broadcast in [[false, false], [false, true], [true, false], [true, true]] {
            verify(&session, &foreign, &left64, &right64, policy, broadcast);
            verify(&session, &foreign, &left32, &right32, policy, broadcast);
        }
    }
}

#[test]
#[ignore = "Requires actual Apple GPU four-low-format backward encoding/status and owner proof."]
fn relu_backward_four_low_formats_all_policies_and_roles() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    macro_rules! low {
        ($ty:ty,$bits:ty,$edges:expr) => {{
            let edges: [$bits; 11] = $edges;
            let mut left = Vec::new();
            let mut right = Vec::new();
            for a in edges {
                for b in edges {
                    left.push(<$ty>::from_bits(a));
                    right.push(<$ty>::from_bits(b));
                }
            }
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                for broadcast in [[false, false], [false, true], [true, false], [true, true]] {
                    verify(&session, &foreign, &left, &right, policy, broadcast);
                }
            }
        }};
    }
    low!(
        fusion_pcu::PcuF16Bits,
        u16,
        [
            0, 0x8000, 1, 0x8001, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0x7c00, 0xfc00, 0x7e01
        ]
    );
    low!(
        fusion_pcu::PcuBf16Bits,
        u16,
        [
            0, 0x8000, 1, 0x8001, 0x0080, 0x3f80, 0xbf80, 0x7f7f, 0x7f80, 0xff80, 0x7fc1
        ]
    );
    low!(
        fusion_pcu::PcuF8E4M3FnBits,
        u8,
        [0, 0x80, 1, 0x81, 8, 0x38, 0xb8, 0x7e, 0x7f, 0xff, 0x78]
    );
    low!(
        fusion_pcu::PcuF8E5M2Bits,
        u8,
        [0, 0x80, 1, 0x81, 4, 0x3c, 0xbc, 0x7b, 0x7c, 0xfc, 0x7f]
    );
}

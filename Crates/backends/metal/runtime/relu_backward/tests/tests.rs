//! Actual encoding/status compared with the independent core finite-selection law.
use super::*;
use fusion_pcu::{PcuCheckedFloat, PcuExecutionFaultKind};
fn bytes<T: PcuCheckedFloat>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn verify<T: PcuCheckedFloat>(
    session: &MetalSession,
    left: &[T],
    right: &[T],
    policy: PcuFloatUnderflowPolicy,
    broadcast: [bool; 2],
) {
    let count = left.len();
    let width = usize::from(T::TYPE.bit_width()) / 8;
    let map = session
        .prepare_relu_backward(T::TYPE, policy, broadcast)
        .unwrap();
    let left_raw = bytes(if broadcast[0] { &left[..1] } else { left });
    let right_raw = bytes(if broadcast[1] { &right[..1] } else { right });
    let input = session.upload_bytes(&left_raw).unwrap();
    let upstream = session.upload_bytes(&right_raw).unwrap();
    let output = session.allocate_zeroed_bytes(count * width).unwrap();
    let records = session.0.native.allocate(count * 4).unwrap();
    records.fill_ones();
    session
        .0
        .native
        .execute(
            &map.pipeline,
            [&input.native, &upstream.native, &output.native, &records],
            [u32::try_from(count).unwrap(), map.profile],
            count,
        )
        .unwrap();
    let mut actual = vec![0; count * width];
    output.read_into_bytes(&mut actual).unwrap();
    let status = records.read(count).unwrap();
    let mut fatal = false;
    for index in 0..count {
        let a = left[if broadcast[0] { 0 } else { index }];
        let b = right[if broadcast[1] { 0 } else { index }];
        match a.pcu_checked_relu_backward_with_policy(b, policy) {
            Ok(value) => {
                assert_eq!(status[index], 0);
                assert_eq!(
                    &actual[index * width..(index + 1) * width],
                    value.encode_le().as_ref()
                );
            }
            Err(kind) => {
                fatal = true;
                assert_eq!(
                    status[index],
                    match kind {
                        PcuExecutionFaultKind::InvalidFloatingOperand => 4,
                        PcuExecutionFaultKind::ArithmeticUnderflow => 3,
                        _ => panic!("selection-only fault"),
                    }
                );
            }
        }
    }
    let completed = map.execute_completed(&input, &upstream, count);
    if fatal {
        assert!(matches!(completed,Err(MetalError::Arithmetic(fault)) if !fault.recovered));
    } else {
        let (owner, notice) = completed.unwrap();
        assert!(notice.is_none());
        let mut retained = vec![0; count * width];
        owner.read_into_bytes(&mut retained).unwrap();
        assert_eq!(retained, actual);
    }
    let foreign = MetalSession::open(0)
        .unwrap()
        .upload_bytes(&right_raw)
        .unwrap();
    assert!(matches!(
        map.execute_completed(&input, &foreign, count),
        Err(MetalError::ForeignSession)
    ));
    if width > 1 {
        let short = session.allocate_zeroed_bytes(width - 1).unwrap();
        assert!(matches!(
            map.execute_completed(&short, &upstream, count),
            Err(MetalError::InvalidExtent)
        ));
    } else {
        assert!(matches!(
            session.allocate_zeroed_bytes(0),
            Err(MetalError::InvalidExtent)
        ));
    }
    let mut original = vec![0; left_raw.len()];
    input.read_into_bytes(&mut original).unwrap();
    assert_eq!(original, left_raw);
}
#[test]
#[ignore = "Requires actual Metal GPU backward-selection oracle."]
fn relu_backward_f32_f64_all_policies_dense_broadcast_payload_status_and_guards() {
    let session = MetalSession::open(0).unwrap();
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
    for _ in 0..4096 {
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
            verify(&session, &left64, &right64, policy, broadcast);
            verify(&session, &left32, &right32, policy, broadcast);
        }
    }
}

#[test]
#[ignore = "Requires actual Apple GPU four-low-format backward encoding/status and owner proof."]
fn relu_backward_four_low_formats_all_policies_and_roles() {
    let session = MetalSession::open(0).unwrap();
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
                    verify(&session, &left, &right, policy, broadcast);
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

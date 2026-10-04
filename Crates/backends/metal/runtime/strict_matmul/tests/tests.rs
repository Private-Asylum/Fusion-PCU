//! Native ordered step oracle and strict dependent-record refusal.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuExecutionFaultKind,
};
#[test]
fn strict_matmul_event_domain_rejects_impossible_status_and_wrong_extent() {
    let domain = TensorStrictFaultDomain::matmul(
        PcuScalarType::F64,
        2,
        3,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    )
    .unwrap();
    assert!(validate(&[0; 4], domain).is_ok());
    assert!(validate(&[0; 3], domain).is_err());
    for records in [
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
        assert!(validate(&records, domain).is_err());
    }
    assert!(
        matches!(validate(&[4,1,6,4],domain),Err(MetalError::Arithmetic(fault)) if fault.invocation_id==4)
    );
}
fn bytes<T: PcuCheckedFloat>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn tag(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 1,
        PcuExecutionFaultKind::ArithmeticUnderflow => 3,
        PcuExecutionFaultKind::InvalidFloatingOperand => 4,
        _ => panic!("dot product fault"),
    }
}
fn verify<T: PcuCheckedFloat>(
    map: &MetalPreparedStrictMatMul,
    left: &[T; 9],
    right: &[T; 6],
    zero: T,
    policy: PcuFloatUnderflowPolicy,
) {
    let session = &map.session;
    let left_bytes = bytes(left);
    let right_bytes = bytes(right);
    let a = session.upload_bytes(&left_bytes).unwrap();
    let b = session.upload_bytes(&right_bytes).unwrap();
    let output = session.allocate_zeroed_bytes(map.output_bytes).unwrap();
    assert_eq!(map.status_word_extent(), 12);
    let records = session
        .0
        .native
        .allocate(map.status_word_extent() * 4)
        .unwrap();
    records.fill_ones();
    session
        .0
        .native
        .execute(
            &map.pipeline,
            [&a.native, &b.native, &output.native, &records],
            [6, map.policy],
            6,
        )
        .unwrap();
    let actual = records.read(map.status_word_extent()).unwrap();
    let mut payload = vec![0; map.output_bytes];
    output.read_into_bytes(&mut payload).unwrap();
    let width = usize::from(T::TYPE.bit_width()) / 8;
    let mut expected = [0; 36];
    for cell in 0..6 {
        let mut accumulator = zero;
        let mut failed = false;
        for reduction in 0..3 {
            let base = cell * 6 + reduction * 2;
            let product = match left[cell / 2 * 3 + reduction]
                .pcu_checked_mul_with_policy(right[reduction * 2 + cell % 2], policy)
            {
                Ok(value) => value,
                Err(kind) => {
                    expected[base] = tag(kind);
                    failed = true;
                    break;
                }
            };
            match accumulator.pcu_checked_add_with_policy(product, policy) {
                Ok(value) => accumulator = value,
                Err(kind) => {
                    expected[base + 1] = tag(kind);
                    failed = true;
                    break;
                }
            }
        }
        if !failed {
            assert_eq!(
                &payload[cell * width..(cell + 1) * width],
                accumulator.encode_le().as_ref()
            );
        }
    }
    let compact: Vec<_> = expected
        .as_chunks::<6>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(cell, statuses)| {
            statuses
                .iter()
                .position(|&status| status != 0)
                .map_or([0, 0], |offset| {
                    [u32::try_from(cell * 6 + offset).unwrap(), statuses[offset]]
                })
        })
        .collect();
    assert_eq!(actual, compact);
    match map.execute_completed(&a, &b) {
        Ok(completed) => {
            assert!(expected.iter().all(|&record| record == 0));
            let mut escaped = vec![0; map.output_bytes];
            completed.read_into_bytes(&mut escaped).unwrap();
            assert_eq!(escaped, payload);
        }
        Err(MetalError::Arithmetic(fault)) => {
            let ordinal = expected.iter().position(|&record| record != 0).unwrap();
            assert_eq!(fault.invocation_id, u64::try_from(ordinal).unwrap());
            assert!(map.domain.accepts(fault));
        }
        Err(error) => panic!("unexpected {error:?}"),
    }
    let mut original = vec![0; left_bytes.len()];
    a.read_into_bytes(&mut original).unwrap();
    assert_eq!(original, left_bytes);
    let mut original = vec![0; right_bytes.len()];
    b.read_into_bytes(&mut original).unwrap();
    assert_eq!(original, right_bytes);
}
fn native<T: PcuCheckedFloat>(
    session: &MetalSession,
    foreign: &MetalSession,
    zero: T,
    edges: &[T],
    from_bits: impl Fn(u64) -> T,
) {
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        let map = session
            .prepare_strict_matmul(T::TYPE, [3, 3, 2], policy)
            .unwrap();
        for offset in 0..edges.len() {
            let left = core::array::from_fn(|index| edges[(offset + index) % edges.len()]);
            let right = core::array::from_fn(|index| edges[(offset + index * 3) % edges.len()]);
            verify(&map, &left, &right, zero, policy);
        }
        let mut seed = 0x739a_93b1_e518_2525u64;
        for _ in 0..128 {
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                from_bits(seed)
            };
            let left = core::array::from_fn(|_| next());
            let right = core::array::from_fn(|_| next());
            verify(&map, &left, &right, zero, policy);
        }
        let good = session.upload_bytes(&bytes(&[zero; 9])).unwrap();
        let other = foreign.upload_bytes(&bytes(&[zero; 6])).unwrap();
        assert!(matches!(
            map.execute_completed(&good, &other),
            Err(MetalError::ForeignSession)
        ));
        let short = session.allocate_zeroed_bytes(1).unwrap();
        assert!(matches!(
            map.execute_completed(&short, &good),
            Err(MetalError::InvalidExtent)
        ));
    }
}
#[test]
#[ignore = "Requires actual native Metal ordered MatMul compilation and terminal execution."]
fn strict_matmul_native_f32_f64_ordered_steps_payload_status_and_owners() {
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
        0f32,
        &bits32.map(f32::from_bits),
        |bits| f32::from_bits(u32::try_from(bits & 0xffff_ffff).unwrap()),
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
        0f64,
        &bits64.map(f64::from_bits),
        f64::from_bits,
    );
    for shape in [[0, 3, 2], [3, 0, 2], [3, 3, 0], [usize::MAX, 3, 2]] {
        assert!(
            session
                .prepare_strict_matmul(
                    PcuScalarType::F64,
                    shape,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding
                )
                .is_err()
        );
    }
}

#[path = "compact/compact.rs"]
mod compact;

//! Compare every private output/status against the independent core integer conversion law.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{PcuClampedFloatConversion,PcuCheckedFloatWidening,PcuClampedError,PcuExecutionFaultKind as Kind};
fn code(kind: Kind) -> u32 {
    match kind {
        Kind::ArithmeticOverflow => 1,
        Kind::ArithmeticUnderflow => 3,
        Kind::InvalidFloatingOperand => 4,
        _ => panic!("conversion-only fault"),
    }
}
fn run(map: &MetalPreparedFloatConversion, words: &[u32], count: usize) -> (Vec<u32>, Vec<u32>) {
    let input = map.session.upload_u32(words).unwrap();
    let width = if map.conversion == PcuDispatchCheckedFloatConversion::F32ToF64 {
        2
    } else {
        1
    };
    let output = map
        .session
        .allocate_zeroed_bytes(count * width * 4)
        .unwrap();
    let records = map.session.0.native.allocate(count * 4).unwrap();
    records.fill_ones();
    map.session
        .0
        .native
        .execute(
            &map.pipeline,
            [&input.native, &input.native, &output.native, &records],
            [u32::try_from(count).unwrap(), map.operation],
            count,
        )
        .unwrap();
    (
        output.native.read(count * width).unwrap(),
        records.read(count).unwrap(),
    )
}
#[test]
#[ignore = "Requires actual macOS Metal device and compiler."]
fn checked_conversion_u32_native_payload_and_status_oracle() {
    let session = MetalSession::open(0).unwrap();
    let mut wide = vec![
        0,
        1,
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0x47ef_ffff_f000_0000,
        0x47ef_ffff_ffff_ffff,
        0x380f_ffff_e000_0000,
        0x380f_ffff_f000_0000,
        0x3690_0000_0000_0000,
        0x36a0_0000_0000_0000,
        0x3810_0000_0000_0000,
    ];
    let mut state = 0x1234_5678_8765_4321_u64;
    for _ in 0..16_384 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        wide.push(state);
    }
    let words: Vec<u32> = wide
        .iter()
        .flat_map(|bits| {
            [
                u32::try_from(bits & 0xffff_ffff).unwrap(),
                u32::try_from(bits >> 32).unwrap(),
            ]
        })
        .collect();
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let map = session
                .prepare_float_conversion(
                    PcuDispatchCheckedFloatConversion::F64ToF32,
                    policy,
                    range,
                    false,
                )
                .unwrap();
            let (payload, status) = run(&map, &words, wide.len());
            for (lane, &bits) in wide.iter().enumerate() {
                let (expected, record) =
                    match f64::from_bits(bits).pcu_clamped_to_f32_with_policy(policy) {
                        Ok(value) => (value.to_bits(), 0),
                        Err(PcuClampedError::Range(fault)) => {
                            let record = code(fault.kind())
                                | if range == PcuRangePolicy::Clamp {
                                    0x100
                                } else {
                                    0
                                };
                            (fault.clamped_value().to_bits(), record)
                        }
                        Err(PcuClampedError::Fatal(kind)) => (0, code(kind)),
                    };
                assert_eq!(status[lane], record, "status {bits:x} {policy:?} {range:?}");
                assert_eq!(
                    payload[lane], expected,
                    "payload {bits:x} {policy:?} {range:?}"
                );
            }
        }
    }
    let narrow: Vec<u32> = words
        .iter()
        .copied()
        .chain([0, 1, 0x8000_0000, 0x7f80_0000, 0x7fc0_0001])
        .collect();
    let map = session
        .prepare_float_conversion(
            PcuDispatchCheckedFloatConversion::F32ToF64,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuRangePolicy::Clamp,
            false,
        )
        .unwrap();
    let (payload, status) = run(&map, &narrow, narrow.len());
    for (lane, &bits) in narrow.iter().enumerate() {
        let (expected, record) = match f32::from_bits(bits).pcu_checked_to_f64() {
            Ok(value) => (value.to_bits(), 0),
            Err(kind) => (0, code(kind)),
        };
        assert_eq!(status[lane], record);
        assert_eq!(
            u64::from(payload[2 * lane]) | u64::from(payload[2 * lane + 1]) << 32,
            expected
        );
    }
}

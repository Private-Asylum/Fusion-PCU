//! Terminal diagnostic snapshots qualify every lane, separately from logical output publication.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuDispatchFloatBinaryOp as Op,
    PcuExecutionFaultKind as Kind,
    PcuHostArgument,
    PcuBindingRef,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
trait Encoding: PcuCheckedFloat {
    fn encoding(value: u16) -> Self;
    fn bits(self) -> u16;
}
macro_rules! encoding {
    ($ty:ty, $word:ty) => {
        impl Encoding for $ty {
            fn encoding(value: u16) -> Self {
                Self::from_bits(<$word>::try_from(value).unwrap())
            }
            fn bits(self) -> u16 {
                u16::from(self.to_bits())
            }
        }
    };
}
encoding!(PcuF16Bits, u16);
encoding!(PcuBf16Bits, u16);
encoding!(PcuF8E4M3FnBits, u8);
encoding!(PcuF8E5M2Bits, u8);
fn oracle<T: Encoding>(op: Op, policy: PcuFloatUnderflowPolicy, a: T, b: T) -> Result<T, Kind> {
    match op {
        Op::Add => a.pcu_checked_add_with_policy(b, policy),
        Op::Sub => a.pcu_checked_sub_with_policy(b, policy),
        Op::Mul => a.pcu_checked_mul_with_policy(b, policy),
        Op::Div => a.pcu_checked_div_with_policy(b, policy),
    }
}
fn diagnostic(kind: Kind) -> u32 {
    match kind {
        Kind::ArithmeticOverflow => 1,
        Kind::DivideByZero => 2,
        Kind::ArithmeticUnderflow => 3,
        Kind::InvalidFloatingOperand => 4,
        Kind::SignedDivisionOverflow => {
            panic!("integer-only fault cannot arise from floating arithmetic")
        }
    }
}
fn qualify<T: Encoding>(left: &[T], right: &[T]) {
    qualify_broadcast(left, right, [false; 2]);
}
fn qualify_broadcast<T: Encoding>(left: &[T], right: &[T], broadcast: [bool; 2]) {
    let logical_count = left.len().max(right.len());
    let session = MetalSession::open(0).unwrap();
    let left_gpu = session
        .upload_bytes(PcuHostArgument::read(PcuBindingRef::new(0, 0), left).bytes())
        .unwrap();
    let right_gpu = session
        .upload_bytes(PcuHostArgument::read(PcuBindingRef::new(0, 1), right).bytes())
        .unwrap();
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
            let map = crate::admission::binary::tests::prepared_portable::<T>(
                &session,
                u32::try_from(logical_count).unwrap(),
                op,
                policy,
                broadcast,
            );
            let map = map.test_map();
            let output = session
                .allocate_zeroed_bytes(logical_count * usize::from(T::TYPE.bit_width()) / 8)
                .unwrap();
            let records = session.0.native.allocate(logical_count * 4).unwrap();
            records.fill_ones();
            session
                .0
                .native
                .execute(
                    &map.pipeline,
                    [
                        &left_gpu.native,
                        &right_gpu.native,
                        &output.native,
                        &records,
                    ],
                    [u32::try_from(logical_count).unwrap(), map.operation],
                    logical_count,
                )
                .unwrap();
            let statuses = records.read(logical_count).unwrap();
            let mut values = vec![T::encoding(0); logical_count];
            let mut host = PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut values);
            output.read_into_bytes(host.bytes_mut().unwrap()).unwrap();
            for (index, (&status, &actual)) in statuses.iter().zip(&values).enumerate() {
                let a = left[if broadcast[0] { 0 } else { index }];
                let b = right[if broadcast[1] { 0 } else { index }];
                let (expected_status, expected_bits) = match oracle(op, policy, a, b) {
                    Ok(value) => (0, value.bits()),
                    Err(kind) => (diagnostic(kind), 0),
                };
                assert_eq!(
                    (status, actual.bits()),
                    (expected_status, expected_bits),
                    "{:?}/{op:?}/{policy:?} lane{index} {:x}/{:x}",
                    T::TYPE,
                    a.bits(),
                    b.bits()
                );
            }
        }
    }
}
fn exhaustive<T: Encoding>() {
    let mut left = Vec::with_capacity(65536);
    let mut right = Vec::with_capacity(65536);
    for a in 0..=255 {
        for b in 0..=255 {
            left.push(T::encoding(a));
            right.push(T::encoding(b));
        }
    }
    qualify(&left, &right);
}
fn edges_random<T: Encoding>(edges: &[u16]) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &a in edges {
        for &b in edges {
            left.push(T::encoding(a));
            right.push(T::encoding(b));
        }
    }
    let mut seed = 0x9e37_79b9_u32;
    for _ in 0..8192 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let bytes = seed.to_le_bytes();
        left.push(T::encoding(u16::from_le_bytes([bytes[0], bytes[1]])));
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let bytes = seed.to_le_bytes();
        right.push(T::encoding(u16::from_le_bytes([bytes[0], bytes[1]])));
    }
    qualify(&left, &right);
}
fn every_encoding_zero_one<T: Encoding>(one: u16) {
    let mut left = Vec::with_capacity(131_072);
    let mut right = Vec::with_capacity(131_072);
    for bits in 0..=u16::MAX {
        for rhs in [0, one] {
            left.push(T::encoding(bits));
            right.push(T::encoding(rhs));
        }
    }
    qualify(&left, &right);
}
#[test]
#[ignore = "Requires actual Metal E4M3FN/E5M2 exhaustive encoding and fault proof."]
fn fp8_all_1572864_operation_policy_encoding_pairs() {
    exhaustive::<PcuF8E4M3FnBits>();
    exhaustive::<PcuF8E5M2Bits>();
}
#[test]
#[ignore = "Requires actual Metal half/bfloat bit and fault proof."]
fn f16_bf16_complete_edges_and_random_bits_faults() {
    every_encoding_zero_one::<PcuF16Bits>(0x3c00);
    every_encoding_zero_one::<PcuBf16Bits>(0x3f80);
    edges_random::<PcuF16Bits>(&[
        0, 0x8000, 1, 2, 3, 0x3ff, 0x400, 0x401, 0x1000, 0x3c00, 0x3c01, 0x3c02, 0x4000, 0x7bff,
        0xfbff, 0x7c00, 0xfc00, 0x7c01, 0x7e00, 0x8001,
    ]);
    edges_random::<PcuBf16Bits>(&[
        0, 0x8000, 1, 2, 3, 0x7f, 0x80, 0x81, 0x3380, 0x3f80, 0x3f81, 0x3f82, 0x4000, 0x7f7f,
        0xff7f, 0x7f80, 0xff80, 0x7f81, 0x7fc0, 0x8001,
    ]);
}

fn all_encoding_broadcasts<T: Encoding>(maximum: u16, scalar_edges: &[u16]) {
    let values: Vec<_> = (0..=maximum).map(T::encoding).collect();
    for &scalar in scalar_edges {
        let scalar = [T::encoding(scalar)];
        qualify_broadcast(&scalar, &values, [true, false]);
        qualify_broadcast(&values, &scalar, [false, true]);
    }
}
#[test]
#[ignore = "Requires actual requested Portable exact-byte broadcast indexing/status proof."]
fn portable_broadcast_every_encoding_five_scalar_edges_both_operand_roles() {
    all_encoding_broadcasts::<PcuF16Bits>(u16::MAX, &[0x4000, 1, 0x7bff, 0x7e00, 0]);
    all_encoding_broadcasts::<PcuBf16Bits>(u16::MAX, &[0x4000, 1, 0x7f7f, 0x7fc0, 0]);
    all_encoding_broadcasts::<PcuF8E4M3FnBits>(255, &[0x40, 1, 0x7e, 0x7f, 0]);
    all_encoding_broadcasts::<PcuF8E5M2Bits>(255, &[0x40, 1, 0x7b, 0x7e, 0]);
}

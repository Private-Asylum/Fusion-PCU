//! Independent core integer-reference bits and per-lane diagnostic proof for all six formats.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuCheckedFloat,
    PcuClampedError,
    PcuClampedFloat,
    PcuExecutionFaultKind as Kind,
    PcuHostArgument,
    PcuRangePolicy as Range,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
type Op = PcuDispatchFloatBinaryOp;
trait Encoding: PcuCheckedFloat + PcuClampedFloat {
    const MASK: u64;
    const ONE: u64;
    const MAXIMUM: u64;
    const NORMAL: u64;
    fn raw(bits: u64) -> Self;
    fn bits(self) -> u64;
}
macro_rules! encoding {
    ($ty:ty, $word:ty, $one:expr, $max:expr, $normal:expr) => {
        impl Encoding for $ty {
            const MASK: u64 = u64::MAX >> (64 - <$word>::BITS);
            const ONE: u64 = $one;
            const MAXIMUM: u64 = $max;
            const NORMAL: u64 = $normal;
            fn raw(bits: u64) -> Self {
                Self::from_bits(<$word>::try_from(bits).unwrap())
            }
            fn bits(self) -> u64 {
                Into::<u64>::into(self.to_bits())
            }
        }
    };
}
encoding!(PcuF16Bits, u16, 0x3c00, 0x7bff, 0x400);
encoding!(PcuBf16Bits, u16, 0x3f80, 0x7f7f, 0x80);
encoding!(PcuF8E4M3FnBits, u8, 0x38, 0x7e, 8);
encoding!(PcuF8E5M2Bits, u8, 0x3c, 0x7b, 4);
encoding!(f32, u32, 0x3f80_0000, 0x7f7f_ffff, 0x80_0000);
encoding!(
    f64,
    u64,
    0x3ff0_0000_0000_0000,
    0x7fef_ffff_ffff_ffff,
    0x10_0000_0000_0000
);
fn status(kind: Kind) -> u32 {
    match kind {
        Kind::ArithmeticOverflow => 1,
        Kind::DivideByZero => 2,
        Kind::ArithmeticUnderflow => 3,
        Kind::InvalidFloatingOperand => 4,
        other @ Kind::SignedDivisionOverflow => panic!("unexpected float fault {other:?}"),
    }
}
fn oracle<T: Encoding>(
    a: T,
    b: T,
    op: Op,
    policy: PcuFloatUnderflowPolicy,
    range: Range,
) -> (u64, u32) {
    let result = match op {
        Op::Add => a.pcu_clamped_add_with_policy(b, policy),
        Op::Sub => a.pcu_clamped_sub_with_policy(b, policy),
        Op::Mul => a.pcu_clamped_mul_with_policy(b, policy),
        Op::Div => a.pcu_clamped_div_with_policy(b, policy),
    };
    match result {
        Ok(value) => (value.bits(), 0),
        Err(PcuClampedError::Range(fault)) => {
            if range == Range::Clamp {
                let diagnostic = status(fault.kind()) | 0x100;
                (fault.clamped_value().bits(), diagnostic)
            } else {
                (0, status(fault.kind()))
            }
        }
        Err(PcuClampedError::Fatal(kind)) => (0, status(kind)),
    }
}
fn qualify<T: Encoding>(a: &[T], b: &[T], broadcast: [bool; 2]) {
    let session = MetalSession::open(0).unwrap();
    let count = a.len().max(b.len());
    let bytes = count * usize::from(T::TYPE.bit_width()) / 8;
    let left = session
        .upload_bytes(PcuHostArgument::read(PcuBindingRef::new(0, 0), a).bytes())
        .unwrap();
    let right = session
        .upload_bytes(PcuHostArgument::read(PcuBindingRef::new(0, 1), b).bytes())
        .unwrap();
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for range in [Range::Reject, Range::Clamp] {
            for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
                let map = session
                    .prepare_checked_float_binary_with_range(T::TYPE, op, policy, range)
                    .unwrap()
                    .with_broadcast(broadcast);
                let output = session.allocate_zeroed_bytes(bytes).unwrap();
                let records = session.0.native.allocate(count * 4).unwrap();
                records.fill_ones();
                session
                    .0
                    .native
                    .execute(
                        &map.pipeline,
                        [&left.native, &right.native, &output.native, &records],
                        [u32::try_from(count).unwrap(), map.operation],
                        count,
                    )
                    .unwrap();
                let statuses = records.read(count).unwrap();
                let mut result = vec![T::raw(0); count];
                let mut destination =
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut result);
                output
                    .read_into_bytes(destination.bytes_mut().unwrap())
                    .unwrap();
                for (index, (&value, diagnostic)) in result.iter().zip(statuses).enumerate() {
                    let lhs = a[if broadcast[0] { 0 } else { index }];
                    let rhs = b[if broadcast[1] { 0 } else { index }];
                    assert_eq!(
                        (value.bits(), diagnostic),
                        oracle(lhs, rhs, op, policy, range),
                        "{:?}/{op:?}/{policy:?}/{range:?}/{broadcast:?} lane{index} {:x}/{:x}",
                        T::TYPE,
                        lhs.bits(),
                        rhs.bits()
                    );
                }
            }
        }
    }
}
fn samples<T: Encoding>() -> (Vec<T>, Vec<T>) {
    let sign = (T::MASK >> 1) + 1;
    let edges = [
        0,
        sign,
        1,
        sign + 1,
        T::NORMAL - 1,
        T::NORMAL,
        T::NORMAL + 1,
        T::ONE - 1,
        T::ONE,
        T::ONE + 1,
        T::MAXIMUM,
        T::MAXIMUM + 1,
        T::MASK,
        T::ONE + sign,
    ];
    let mut left = Vec::new();
    let mut right = Vec::new();
    for a in edges {
        for b in edges {
            left.push(T::raw(a));
            right.push(T::raw(b));
        }
    }
    let mut state = 0xa076_1d64_78bd_642f_u64;
    while left.len() < 131_584 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        left.push(T::raw(state & T::MASK));
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        right.push(T::raw(state & T::MASK));
    }
    (left, right)
}
fn low<T: Encoding>() {
    let mut left = Vec::new();
    let mut right = Vec::new();
    if T::MASK == 255 {
        for a in 0..=255 {
            for b in 0..=255 {
                left.push(T::raw(a));
                right.push(T::raw(b));
            }
        }
    } else {
        for a in 0..=65535 {
            for b in [0, T::ONE, T::MAXIMUM, 1] {
                left.push(T::raw(a));
                right.push(T::raw(b));
            }
        }
    }
    qualify(&left, &right, [false; 2]);
    let values: Vec<_> = (0..=T::MASK).map(T::raw).collect();
    for bits in [0, 1, T::ONE, T::MAXIMUM, T::MASK] {
        let scalar = [T::raw(bits)];
        qualify(&scalar, &values, [true, false]);
        qualify(&values, &scalar, [false, true]);
    }
}
#[test]
#[ignore = "Requires actual M4 low4 Clamp/Reject complete encoding/status and broadcast proof."]
fn low4_binary_range_all_encodings_and_broadcast_status() {
    low::<PcuF16Bits>();
    low::<PcuBf16Bits>();
    low::<PcuF8E4M3FnBits>();
    low::<PcuF8E5M2Bits>();
}
fn native<T: Encoding>() {
    let (left, right) = samples::<T>();
    qualify(&left, &right, [false; 2]);
    for bits in [0, 1, T::ONE, T::MAXIMUM, T::MASK] {
        let scalar = [T::raw(bits)];
        qualify(&scalar, &left, [true, false]);
        qualify(&left, &scalar, [false, true]);
    }
}
#[test]
#[ignore = "Requires actual M4 native-width stratified/random Clamp/Reject bits/fault proof."]
fn f32_f64_binary_range_stratified_pairs_and_broadcast_status() {
    native::<f32>();
    native::<f64>();
}

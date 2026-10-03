//! Independent stratified/random native-width encoding oracle, not an exhaustive F32/F64 claim.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuCheckedFloat,
    PcuClampedFloat,
    PcuClampedError,
    PcuExecutionFaultKind,
    PcuHostArgument,
};
trait Encoding: PcuCheckedFloat + PcuClampedFloat {
    const SIGN: u64;
    const MAXIMUM: u64;
    const NORMAL: u64;
    const MASK: u64;
    fn raw(bits: u64) -> Self;
    fn bits(self) -> u64;
}
impl Encoding for f32 {
    const SIGN: u64 = 0x8000_0000;
    const MAXIMUM: u64 = 0x7f7f_ffff;
    const NORMAL: u64 = 0x0080_0000;
    const MASK: u64 = 0xffff_ffff;
    fn raw(bits: u64) -> Self {
        Self::from_bits(u32::try_from(bits).unwrap())
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}
impl Encoding for f64 {
    const SIGN: u64 = 0x8000_0000_0000_0000;
    const MAXIMUM: u64 = 0x7fef_ffff_ffff_ffff;
    const NORMAL: u64 = 0x0010_0000_0000_0000;
    const MASK: u64 = u64::MAX;
    fn raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}
fn samples<T: Encoding>() -> Vec<T> {
    let mut values = Vec::with_capacity(131_584);
    for magnitude in [
        0,
        1,
        2,
        T::NORMAL - 1,
        T::NORMAL,
        T::NORMAL + 1,
        T::MAXIMUM - 1,
        T::MAXIMUM,
        T::MAXIMUM + 1,
        T::SIGN - 1,
    ] {
        values.push(T::raw(magnitude));
        values.push(T::raw(magnitude + T::SIGN));
    }
    for bit in 0..T::TYPE.bit_width() - 1 {
        let magnitude = 1_u64 << bit;
        values.push(T::raw(magnitude));
        values.push(T::raw(magnitude + T::SIGN));
        values.push(T::raw((magnitude - 1) | T::SIGN));
    }
    let mut state = 0xa076_1d64_78bd_642f_u64;
    while values.len() < 131_584 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(T::raw(state & T::MASK));
    }
    values
}
fn diagnostic(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::InvalidFloatingOperand => 4,
        PcuExecutionFaultKind::ArithmeticUnderflow => 3,
        other => panic!("exact unary cannot produce {other:?}"),
    }
}
fn independent<T: Encoding>(
    bits: u64,
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
) -> Result<(u64, bool), PcuExecutionFaultKind> {
    let (sign, maximum, normal) = (T::SIGN, T::MAXIMUM, T::NORMAL);
    let negative = bits >= sign;
    let magnitude = if negative { bits - sign } else { bits };
    if magnitude > maximum {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    let output = match op {
        PcuDispatchFloatUnaryOp::Neg => {
            if negative {
                magnitude
            } else {
                magnitude + sign
            }
        }
        PcuDispatchFloatUnaryOp::Relu => {
            if negative {
                0
            } else {
                magnitude
            }
        }
    };
    let output_magnitude = if output >= sign {
        output - sign
    } else {
        output
    };
    Ok((
        output,
        policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
            && (1..normal).contains(&output_magnitude),
    ))
}
fn qualify<T: Encoding>() {
    let input = samples::<T>();
    let count = input.len();
    let session = MetalSession::open(0).unwrap();
    let host = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input);
    let device = session.upload_bytes(host.bytes()).unwrap();
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for op in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let map = session
                    .prepare_checked_float_unary_with_range(T::TYPE, op, policy, range)
                    .unwrap();
                let output = session.allocate_zeroed_bytes(host.bytes().len()).unwrap();
                let records = session.0.native.allocate(count * 4).unwrap();
                records.fill_ones();
                session
                    .0
                    .native
                    .execute(
                        &map.pipeline,
                        [&device.native, &device.native, &output.native, &records],
                        [u32::try_from(count).unwrap(), map.operation],
                        count,
                    )
                    .unwrap();
                let statuses = records.read(count).unwrap();
                let mut values = vec![T::raw(0); count];
                let mut destination =
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut values);
                output
                    .read_into_bytes(destination.bytes_mut().unwrap())
                    .unwrap();
                for (index, ((&source, &value), status)) in
                    input.iter().zip(&values).zip(statuses).enumerate()
                {
                    let core = match op {
                        PcuDispatchFloatUnaryOp::Neg => source.pcu_clamped_neg_with_policy(policy),
                        PcuDispatchFloatUnaryOp::Relu => {
                            source.pcu_clamped_relu_with_policy(policy)
                        }
                    };
                    let expected = independent::<T>(source.bits(), op, policy);
                    let core = match core {
                        Ok(value) => Ok((value.bits(), false)),
                        Err(PcuClampedError::Range(fault)) => {
                            assert_eq!(fault.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
                            Ok((fault.clamped_value().bits(), true))
                        }
                        Err(PcuClampedError::Fatal(kind)) => Err(kind),
                    };
                    assert_eq!(
                        core,
                        expected,
                        "reference {:?}/{op:?}/{policy:?}/{range:?} bits{index:x}",
                        T::TYPE
                    );
                    let pair = match expected {
                        Ok((bits, false)) => (0, bits),
                        Ok((bits, true)) if range == PcuRangePolicy::Clamp => (0x103, bits),
                        Ok((_, true)) => (3, 0),
                        Err(kind) => (diagnostic(kind), 0),
                    };
                    assert_eq!(
                        (status, value.bits()),
                        pair,
                        "GPU {:?}/{op:?}/{policy:?} bits{index:x}",
                        T::TYPE
                    );
                }
            }
        }
    }
}
macro_rules! proof {
    ($name:ident,$ty:ty) => {
        #[test]
        #[ignore="Requires real Metal; stratified and deterministic random bits are compared against core and independent encoding oracle."]
        fn $name() { qualify::<$ty>(); }
    };
}
proof!(stratified_f32_unary_encodings, f32);
proof!(stratified_f64_unary_encodings, f64);

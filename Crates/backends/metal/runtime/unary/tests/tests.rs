//! Private terminal diagnostics compare every encoding; failed logical outputs are never published.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuCheckedFloat,
    PcuClampedFloat,
    PcuClampedError,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
trait Encoding: PcuCheckedFloat + PcuClampedFloat {
    fn raw(bits: u16) -> Self;
    fn bits(self) -> u16;
}
macro_rules! encoding {
    ($ty:ty,$word:ty) => {
        impl Encoding for $ty {
            fn raw(bits: u16) -> Self {
                Self::from_bits(<$word>::try_from(bits).unwrap())
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
fn diagnostic(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::InvalidFloatingOperand => 4,
        PcuExecutionFaultKind::ArithmeticUnderflow => 3,
        other => panic!("exact unary cannot produce {other:?}"),
    }
}
fn independent<T: Encoding>(
    bits: u16,
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
) -> Result<(u16, bool), PcuExecutionFaultKind> {
    let (sign, maximum, normal) = match T::TYPE {
        PcuScalarType::F16 => (0x8000, 0x7bff, 0x400),
        PcuScalarType::BF16 => (0x8000, 0x7f7f, 0x80),
        PcuScalarType::F8E4M3FN => (0x80, 0x7e, 8),
        PcuScalarType::F8E5M2 => (0x80, 0x7b, 4),
        _ => unreachable!(),
    };
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
    let count = 1usize << T::TYPE.bit_width();
    let input: Vec<T> = (0..count)
        .map(|bits| T::raw(u16::try_from(bits).unwrap()))
        .collect();
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
                    .prepare_low_precision_unary_with_range(T::TYPE, op, policy, range)
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
        #[ignore="Requires real Metal; every encoding is compared against core and independent bit oracle."]
        fn $name() { qualify::<$ty>(); }
    };
}
proof!(complete_f16_unary_encodings, PcuF16Bits);
proof!(complete_bf16_unary_encodings, PcuBf16Bits);
proof!(complete_e4m3fn_unary_encodings, PcuF8E4M3FnBits);
proof!(complete_e5m2_unary_encodings, PcuF8E5M2Bits);

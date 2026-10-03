//! Independent core integer arithmetic oracle for every newly emitted MLX binary shader.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{PcuBindingRef,PcuCheckedFloat,PcuClampedError,PcuClampedFloat,
    PcuExecutionFaultKind as Kind,PcuHostArgument,PcuRangePolicy as Range,
    PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,
    PcuDispatchFloatBinaryOp,PcuFloatUnderflowPolicy};
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
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let native = session.native_for_binary_proof();
    let count = a.len().max(b.len());
    let left = PcuHostArgument::read(PcuBindingRef::new(0, 0), a);
    let right = PcuHostArgument::read(PcuBindingRef::new(0, 1), b);
    let left = EncodedArray::upload(native, T::TYPE, a.len(), left.bytes()).unwrap();
    let right = EncodedArray::upload(native, T::TYPE, b.len(), right.bytes()).unwrap();
    let left_holder = left.clone_holder().unwrap();
    let right_holder = right.clone_holder().unwrap();
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for range in [Range::Reject, Range::Clamp] {
            for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
                let prepared = CheckedBinary::prepare(
                    native,
                    T::TYPE,
                    op,
                    policy,
                    range,
                    count,
                    [a.len(), b.len()],
                    broadcast,
                )
                .unwrap();
                let api = &native.0.api;
                let mut output = Owner::empty(Rc::clone(api), api.array_free);
                let mut records = Owner::empty(Rc::clone(api), api.array_free);
                // SAFETY: exact native prepared primitive and retained typed inputs; distinct sibling holders.
                api.status(|| unsafe {
                    (api.binary_apply)(
                        &raw mut output.raw,
                        &raw mut records.raw,
                        prepared.owner.as_ref().unwrap().raw,
                        left_holder.raw,
                        right_holder.raw,
                    )
                })
                .unwrap();
                validate(
                    native,
                    output.raw,
                    prepared.dtype,
                    prepared.carrier_count,
                    prepared.carrier_width,
                )
                .unwrap();
                validate(native, records.raw, 3, count, 4).unwrap();
                // SAFETY: proof holds both real inputs, sibling outputs and explicit session through terminal wait.
                api.status(|| unsafe { (api.array_eval)(output.raw) })
                    .unwrap();
                api.status(|| unsafe { (api.array_eval)(records.raw) })
                    .unwrap();
                api.status(|| unsafe { (api.synchronize)(native.0.stream.raw) })
                    .unwrap();
                api.status(|| unsafe { (api.array_wait)(output.raw) })
                    .unwrap();
                api.status(|| unsafe { (api.array_wait)(records.raw) })
                    .unwrap();
                available(native, output.raw).unwrap();
                available(native, records.raw).unwrap();
                // SAFETY: initialized terminal dense count UInt32 records holder remains retained during scan.
                let pointer = api
                    .guarded(|| unsafe { (api.array_data_u32)(records.raw) })
                    .unwrap();
                assert!(!pointer.is_null());
                // SAFETY: exact previously validated terminal dense UInt32 extent.
                let statuses = unsafe { std::slice::from_raw_parts(pointer, count) };
                let output = EncodedArray::from_owner(native, T::TYPE, count, output).unwrap();
                let mut result = vec![T::raw(0); count];
                let mut destination =
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut result);
                output.read(destination.bytes_mut().unwrap()).unwrap();
                for (index, (&value, &diagnostic)) in result.iter().zip(statuses).enumerate() {
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
#[ignore = "Requires actual MLX-owned emitted binary UInt kernel and every output/status bit."]
fn native_low_four_binary_bit_and_fault_oracle() {
    low::<PcuF16Bits>();
    low::<PcuBf16Bits>();
    low::<PcuF8E4M3FnBits>();
    low::<PcuF8E5M2Bits>();
}
#[test]
#[ignore = "Requires actual MLX-owned binary kernel sampled half pair edge/random proof."]
fn native_low_binary_pair_edges_and_random() {
    let (a, b) = samples::<PcuF16Bits>();
    qualify(&a, &b, [false; 2]);
    let (a, b) = samples::<PcuBf16Bits>();
    qualify(&a, &b, [false; 2]);
}
#[test]
fn fatal_precedence_and_unknown_status() {
    assert_eq!(
        select_fault(&[0x101, 0x103, 2, 4]).unwrap().unwrap(),
        PcuExecutionFault {
            invocation_id: 2,
            kind: Kind::DivideByZero,
            recovered: false
        }
    );
    assert_eq!(
        select_fault(&[0, 0x103, 0x101])
            .unwrap()
            .unwrap()
            .invocation_id,
        1
    );
    assert!(select_fault(&[0, 0x102]).is_err());
    assert!(select_fault(&[u32::MAX]).is_err());
}

#[test]
#[ignore = "Requires independently emitted actual MLX F32/F64 integer kernels and all payload/status bits."]
fn native_f32_f64_binary_bit_and_fault_oracle() {
    let (a, b) = samples::<f32>();
    qualify(&a, &b, [false; 2]);
    for bits in [
        0,
        1,
        <f32 as Encoding>::ONE,
        <f32 as Encoding>::MAXIMUM,
        <f32 as Encoding>::MASK,
    ] {
        qualify(&[f32::raw(bits)], &b[..1024], [true, false]);
        qualify(&a[..1024], &[f32::raw(bits)], [false, true]);
    }
    let (a, b) = samples::<f64>();
    qualify(&a, &b, [false; 2]);
    for bits in [
        0,
        1,
        <f64 as Encoding>::ONE,
        <f64 as Encoding>::MAXIMUM,
        <f64 as Encoding>::MASK,
    ] {
        qualify(&[f64::raw(bits)], &b[..1024], [true, false]);
        qualify(&a[..1024], &[f64::raw(bits)], [false, true]);
    }
}

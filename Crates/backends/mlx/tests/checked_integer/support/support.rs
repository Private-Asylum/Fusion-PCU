//! Exact integer encoding banks and independent checked/clamped core expectations.
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedInteger,
    PcuScalar,
    PcuScalarType,
    PcuExecutionFault,
    PcuDispatchIntegerBinaryOp as Op,
    PcuRangePolicy as Range,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
pub trait Sample: PcuCheckedInteger {
    fn raw(bytes: &[u8]) -> Self;
    fn zero() -> Self {
        Self::raw(&vec![0; Self::ENCODED_SIZE])
    }
    fn small(value: u8) -> Self {
        let mut bytes = vec![0; Self::ENCODED_SIZE];
        bytes[0] = value;
        Self::raw(&bytes)
    }
    fn max() -> Self {
        let mut bytes = vec![0xff; Self::ENCODED_SIZE];
        if signed(Self::TYPE) {
            *bytes.last_mut().unwrap() = 0x7f;
        }
        Self::raw(&bytes)
    }
    fn min() -> Self {
        let mut bytes = vec![0; Self::ENCODED_SIZE];
        if signed(Self::TYPE) {
            *bytes.last_mut().unwrap() = 0x80;
        }
        Self::raw(&bytes)
    }
}
const fn signed(scalar: PcuScalarType) -> bool {
    matches!(
        scalar,
        PcuScalarType::I8
            | PcuScalarType::I16
            | PcuScalarType::I32
            | PcuScalarType::I64
            | PcuScalarType::I128
            | PcuScalarType::I256
            | PcuScalarType::I512
    )
}
macro_rules! samples { ($($ty:ty=>$width:literal;)+) => {$(impl Sample for $ty {fn raw(bytes:&[u8])->Self { Self::decode_le(<[u8;$width]>::try_from(bytes).unwrap()) }})+}; }
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
pub fn same<T: Sample>(actual: &[T], wanted: &[T]) {
    assert_eq!(actual.len(), wanted.len());
    for (a, b) in actual.iter().zip(wanted) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
pub fn expected<T: Sample>(
    left: &[T],
    right: &[T],
    op: Op,
    range: Range,
    broadcast: [bool; 2],
) -> (Vec<T>, Option<PcuExecutionFault>) {
    let mut fault = None;
    let output = (0..5)
        .map(|index| {
            let a = left[if broadcast[0] { 0 } else { index }];
            let b = right[if broadcast[1] { 0 } else { index }];
            let result = match op {
                Op::Add => a.pcu_clamped_add(b),
                Op::Sub => a.pcu_clamped_sub(b),
                Op::Mul => a.pcu_clamped_mul(b),
            };
            match result {
                Ok(value) => value,
                Err(error) => {
                    fault.get_or_insert_with(|| PcuExecutionFault {
                        invocation_id: index as u64,
                        kind: error.kind(),
                        recovered: range == Range::Clamp,
                    });
                    error.clamped_value()
                }
            }
        })
        .collect();
    (output, fault)
}

//! Independent native primitive checked arithmetic; complete transaction and recovered payload law.
#[rustfmt::skip]
use pcu_facade::{PcuCheckedInteger,PcuExecutionFault,PcuExecutionFaultKind,PcuDispatchIntegerBinaryOp,PcuRangePolicy};
pub trait Native: PcuCheckedInteger + core::fmt::Debug + Eq {
    fn small(value: u8) -> Self;
    fn minimum() -> Self;
    fn maximum() -> Self;
    fn evaluate(
        self,
        right: Self,
        op: PcuDispatchIntegerBinaryOp,
    ) -> Result<Self, PcuExecutionFaultKind>;
}
macro_rules! primitive {
    ($ty:ty) => {
        impl Native for $ty {
            fn small(value: u8) -> Self {
                Self::try_from(value).unwrap()
            }
            fn minimum() -> Self {
                Self::MIN
            }
            fn maximum() -> Self {
                Self::MAX
            }
            fn evaluate(
                self,
                right: Self,
                op: PcuDispatchIntegerBinaryOp,
            ) -> Result<Self, PcuExecutionFaultKind> {
                let result = match op {
                    PcuDispatchIntegerBinaryOp::Add => self.checked_add(right),
                    PcuDispatchIntegerBinaryOp::Sub => self.checked_sub(right),
                    PcuDispatchIntegerBinaryOp::Mul => {
                        unreachable!("bounded Add/Sub native control")
                    }
                };
                result.ok_or_else(|| {
                    let left = i128::from(self);
                    let right = i128::from(right);
                    let result = if op == PcuDispatchIntegerBinaryOp::Add {
                        left + right
                    } else {
                        left - right
                    };
                    if result < i128::from(Self::MIN) {
                        PcuExecutionFaultKind::ArithmeticUnderflow
                    } else {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    }
                })
            }
        }
    };
}
primitive!(i8);
primitive!(u8);
primitive!(i16);
primitive!(u16);
primitive!(i32);
primitive!(u32);
primitive!(i64);
primitive!(u64);
pub fn execute<T: Native, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
) -> Result<(), PcuExecutionFault> {
    assert!(left.len() >= N && right.len() >= N && output.len() >= N);
    if range == PcuRangePolicy::Reject {
        for lane in 0..N {
            left[lane]
                .evaluate(right[lane], op)
                .map_err(|kind| PcuExecutionFault {
                    kind,
                    invocation_id: u64::try_from(lane).unwrap(),
                    recovered: false,
                })?;
        }
        for lane in 0..N {
            output[lane] = left[lane]
                .evaluate(right[lane], op)
                .expect("complete checked transaction preflight");
        }
        return Ok(());
    }
    let mut first = None;
    for lane in 0..N {
        output[lane] = match left[lane].evaluate(right[lane], op) {
            Ok(value) => value,
            Err(kind) => {
                first.get_or_insert_with(|| PcuExecutionFault {
                    kind,
                    invocation_id: u64::try_from(lane).unwrap(),
                    recovered: true,
                });
                if kind == PcuExecutionFaultKind::ArithmeticUnderflow {
                    T::minimum()
                } else {
                    T::maximum()
                }
            }
        };
    }
    first.map_or(Ok(()), Err)
}

//! Shared PCU integer-backed reference oracle checks the unchanged arithmetic under new shapes.
#[rustfmt::skip]
use pcu_facade::{
    PcuClampedError,
    PcuClampedFloat,
    PcuExecutionFault,
};
#[rustfmt::skip]
use super::{
    Op,
    Policy,
    Range,
    Sample,
};
pub fn compare<T: Sample>(left: &[T], right: &[T]) {
    assert_eq!(left.len(), right.len());
    // Byte equality retains every sign bit, including negative zero and unread NaN payloads.
    let left = pcu_facade::PcuHostArgument::read(pcu_facade::PcuBindingRef::new(0, 0), left);
    let right = pcu_facade::PcuHostArgument::read(pcu_facade::PcuBindingRef::new(0, 0), right);
    assert_eq!(left.bytes(), right.bytes());
}
pub fn bank<T: Sample>(phase: usize) -> (Vec<T>, Vec<T>) {
    let mut left = vec![T::from_bits(T::SIGN - 1); 8];
    let mut right = vec![T::from_bits(T::SIGN - 1); 11];
    left[..5].fill(T::value(2.0));
    right[..5].fill(T::value(4.0));
    match phase {
        0 => {}
        1 => {
            left[0] = T::from_bits(1);
            right[0] = T::value(0.5);
        }
        2 => {
            left[0] = T::from_bits(T::MAX);
            right[0] = T::from_bits(T::MAX);
        }
        3 => {
            left[0] = T::from_bits(1);
            right[0] = T::value(0.5);
            left[2] = T::from_bits(T::SIGN - 1);
            right[3] = T::value(0.0);
        }
        4 => {
            right[3] = T::value(0.0);
        }
        5 => {
            left[0] = T::from_bits(1);
            right[0] = T::value(0.0);
        }
        6 => {
            left[0] = T::from_bits(T::MAX);
            right[0] = T::from_bits(T::SIGN | T::MAX);
        }
        7 => {
            left[0] = T::from_bits(T::MAX);
            right[0] = T::value(0.5);
            left[1] = T::from_bits(1);
        }
        _ => unreachable!(),
    }
    (left, right)
}
pub fn expected<T: Sample + PcuClampedFloat>(
    left: &[T],
    right: &[T],
    op: Op,
    policy: Policy,
    range: Range,
    broadcast: [bool; 2],
) -> (Vec<T>, Option<PcuExecutionFault>) {
    let mut output = Vec::with_capacity(5);
    let mut fatal = None;
    let mut recovered = None;
    for lane in 0..5 {
        let a = left[if broadcast[0] { 0 } else { lane }];
        let b = right[if broadcast[1] { 0 } else { lane }];
        let result = match op {
            Op::Add => a.pcu_clamped_add_with_policy(b, policy),
            Op::Sub => a.pcu_clamped_sub_with_policy(b, policy),
            Op::Mul => a.pcu_clamped_mul_with_policy(b, policy),
            Op::Div => a.pcu_clamped_div_with_policy(b, policy),
        };
        match result {
            Ok(value) => output.push(value),
            Err(error) => {
                let fault = PcuExecutionFault {
                    invocation_id: u64::try_from(lane).unwrap(),
                    kind: error.kind(),
                    recovered: range == Range::Clamp && matches!(error, PcuClampedError::Range(_)),
                };
                if fault.recovered {
                    recovered.get_or_insert(fault);
                } else {
                    fatal.get_or_insert(fault);
                }
                output.push(match error {
                    PcuClampedError::Range(range) => range.clamped_value(),
                    PcuClampedError::Fatal(_) => T::value(0.0),
                });
            }
        }
    }
    (output, fatal.or(recovered))
}

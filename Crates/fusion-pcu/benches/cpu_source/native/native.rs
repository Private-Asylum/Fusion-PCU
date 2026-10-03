//! Matched checked, transactional synchronous native controls.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuCheckedInteger,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
#[path = "composition/composition.rs"]
pub mod composition;
#[path = "helper_integer/helper_integer.rs"]
pub mod helper_integer;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Schema,
    Fault(PcuExecutionFault),
}
fn fault(invocation: usize, kind: PcuExecutionFaultKind) -> Error {
    Error::Fault(PcuExecutionFault {
        recovered: false,
        invocation_id: u64::try_from(invocation).unwrap(),
        kind,
    })
}
pub fn negate<const N: usize>(input: &[f32; N], output: &mut [f32]) -> Result<(), Error> {
    if output.len() < N {
        return Err(Error::Schema);
    }
    for (invocation, value) in input.iter().copied().enumerate() {
        value
            .pcu_checked_neg()
            .map_err(|kind| fault(invocation, kind))?;
    }
    for (value, result) in input.iter().zip(output) {
        *result = f32::from_bits(value.to_bits() ^ 0x8000_0000);
    }
    Ok(())
}
pub fn integer<const N: usize, const MUL: bool>(
    lhs: &[u64; N],
    rhs: &[u64; N],
    output: &mut [u64],
) -> Result<(), Error> {
    if output.len() < N {
        return Err(Error::Schema);
    }
    for (invocation, (left, right)) in lhs.iter().copied().zip(rhs.iter().copied()).enumerate() {
        evaluate::<MUL>(left, right).map_err(|kind| fault(invocation, kind))?;
    }
    for ((left, right), result) in lhs.iter().copied().zip(rhs.iter().copied()).zip(output) {
        *result = evaluate::<MUL>(left, right).expect("successful checked preflight");
    }
    Ok(())
}
fn evaluate<const MUL: bool>(left: u64, right: u64) -> Result<u64, PcuExecutionFaultKind> {
    if MUL {
        left.pcu_checked_mul(right)
    } else {
        left.pcu_checked_add(right)
    }
}

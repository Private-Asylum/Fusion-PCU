//! Actual Portable joint source, immutable input borrow, both-output rollback and retry.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuCheckedIntegerDivision,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuScalar,
    PcuTensor,
};
#[pcu(crate_path=::pcu_facade)]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(invocations=5,flag(strict),flag(deterministic),crate_path=::pcu_facade)]
fn divide<T: PcuCheckedIntegerDivision>(left: &[T], q: &mut [T], right: &[T], r: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(left[id], right[id]);
    r[id] = remainder;
    q[id] = quotient;
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })?;
    let left = retain(&[i128::MIN, 17, -17, 23, 5])?;
    let right = retain(&[1_i128, 5, 5, 7, 1])?;
    let mut q = [-91_i128; 7];
    let mut r = [-92_i128; 9];
    divide::<i128>(&left, &mut q, &right, &mut r)?;
    assert_eq!(q, [i128::MIN, 3, -3, 3, 5, -91, -91]);
    assert_eq!(r, [0, 2, -2, 2, 0, -92, -92, -92, -92]);
    let before = (q, r);
    let bad = retain(&[-1_i128, 5, 0, 7, 1])?;
    let error = divide::<i128>(&left, &mut q, &bad, &mut r).unwrap_err();
    let fault = error.arithmetic_fault().ok_or("missing arithmetic fault")?;
    assert_eq!(fault.kind, PcuExecutionFaultKind::SignedDivisionOverflow);
    assert_eq!(fault.invocation_id, 0);
    assert!(!fault.recovered);
    assert_eq!((q, r), before);
    divide::<i128>(&left, &mut q, &right, &mut r)?;
    assert_eq!((q, r), before);
    global::clear_thread_cache()?;
    let mut input = [0_i128; 7];
    left.read_into(&mut input)?;
    assert_eq!(input, [i128::MIN, 17, -17, 23, 5, 0, 0]);
    println!(
        "Portable joint quotient/remainder retained exact i128 bits and both tails; fatal division rolled back both outputs."
    );
    Ok(())
}

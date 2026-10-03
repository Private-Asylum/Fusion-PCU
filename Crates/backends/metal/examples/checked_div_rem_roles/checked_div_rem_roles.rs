//! Genuine requested Portable source through the separately prepared actual-read Metal backend.
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalError,
    MetalSession,
};
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedIntegerDivision,
    PcuExecutionFaultKind,
    PcuHostDispatchError,
};
#[pcu(invocations=5,flag(strict),flag(deterministic),crate_path=::pcu_facade)]
fn divide<T: PcuCheckedIntegerDivision>(
    quotient: &mut [T],
    left: &[T],
    remainder: &mut [T],
    right: &[T],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(left[id], right[id]);
    remainder[id] = r;
    quotient[id] = q;
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = MetalSession::open(0)?;
    let backend = session.checked_div_rem_role_backend();
    let mut call = divide_prepare::<i128, _>(&backend).map_err(|error| format!("{error:?}"))?;
    let left = [i128::MIN, 17, -17, 23, 5];
    let right = [1_i128, 5, 5, 7, 1];
    let mut q = [-91_i128; 7];
    let mut r = [-92_i128; 9];
    call(&mut q, &left, &mut r, &right).map_err(|error| format!("{error:?}"))?;
    assert_eq!(q, [i128::MIN, 3, -3, 3, 5, -91, -91]);
    assert_eq!(r, [0, 2, -2, 2, 0, -92, -92, -92, -92]);
    let before = (q, r);
    let bad = [-1_i128, 5, 0, 7, 1];
    let PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)) =
        call(&mut q, &left, &mut r, &bad).unwrap_err()
    else {
        return Err("missing signed division overflow".into());
    };
    assert_eq!(fault.kind, PcuExecutionFaultKind::SignedDivisionOverflow);
    assert_eq!(fault.invocation_id, 0);
    assert!(!fault.recovered);
    assert_eq!((q, r), before);
    call(&mut q, &left, &mut r, &right).map_err(|error| format!("{error:?}"))?;
    assert_eq!((q, r), before);
    println!(
        "Prepared Portable Metal source retained exact i128 quotient/remainder and both tails; fatal division rolled back both host outputs."
    );
    Ok(())
}

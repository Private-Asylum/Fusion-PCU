//! Genuine annotated primitive quotient/remainder, with joint rollback and retry.
use fusion_pcu_metal::MetalSession;
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedIntegerDivision,
};
#[pcu(invocations=3,flag(strict),crate_path=::pcu_facade)]
fn quotient_remainder<T: PcuCheckedIntegerDivision>(
    left: &[T],
    right: &[T],
    q: &mut [T],
    r: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (a, b) = pcu::checked_div_rem(left[id], right[id]);
    q[id] = a;
    r[id] = b;
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cfg!(target_os = "macos") {
        println!("Requires actual Metal GPU");
        return Ok(());
    }
    let session = MetalSession::open(0)?;
    let backend = session.checked_div_rem_backend();
    let mut call = quotient_remainder_prepare::<i128, _>(&backend)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let mut q = [91; 5];
    let mut r = [93; 5];
    call(&[-7, 7, i128::MIN], &[3, 3, 1], &mut q, &mut r)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    assert_eq!(q, [-2, 2, i128::MIN, 91, 91]);
    assert_eq!(r, [-1, 1, 0, 93, 93]);
    let before = (q, r);
    assert!(call(&[-7, 7, i128::MIN], &[3, 0, -1], &mut q, &mut r).is_err());
    assert_eq!((q, r), before);
    call(&[-7, 7, i128::MIN], &[3, 3, 1], &mut q, &mut r)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    println!("quotient={q:?}, remainder={r:?}");
    Ok(())
}

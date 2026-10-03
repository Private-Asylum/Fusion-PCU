//! Real annotated quotient/remainder preparation; both caller prefixes publish jointly.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedIntegerDivision,
};
#[pcu(invocations=N,crate_path=::pcu_facade)]
fn quotient_remainder<T: PcuCheckedIntegerDivision, const N: usize>(
    left: &[T],
    right: &[T],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(left[id], right[id]);
    quotient[id] = q;
    remainder[id] = r;
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = fusion_pcu_mlx::MlxRuntime::load_default()?;
    let session = runtime.open_gpu(0)?;
    let mut call = quotient_remainder_prepare::<i128, 3, _>(&session.checked_div_rem_backend())
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let mut quotient = [91; 5];
    let mut remainder = [91; 5];
    call(
        &[-23, 23, i128::MIN],
        &[5, -5, 1],
        &mut quotient,
        &mut remainder,
    )
    .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    assert_eq!(quotient, [-4, -4, i128::MIN, 91, 91]);
    assert_eq!(remainder, [-3, 3, 0, 91, 91]);
    let original = (quotient, remainder);
    assert!(
        call(
            &[-23, 23, i128::MIN],
            &[5, 0, 1],
            &mut quotient,
            &mut remainder
        )
        .is_err()
    );
    assert_eq!((quotient, remainder), original);
    println!("MLX completed joint quotient {quotient:?}, remainder {remainder:?}");
    Ok(())
}

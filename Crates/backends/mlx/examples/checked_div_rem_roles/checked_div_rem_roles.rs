//! Actual source role preparation and ordinary execution publish the same two caller prefixes.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuCheckedIntegerDivision,
};

#[pcu(invocations=3,flag(strict),crate_path=::pcu_facade)]
fn divide_by_first<T: PcuCheckedIntegerDivision>(
    unused: &[T],
    remainder: &mut [T],
    input: &[T],
    quotient: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(input[id], input[0]);
    remainder[id] = r;
    quotient[id] = q;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = fusion_pcu_mlx::MlxRuntime::load_default()?;
    let session = runtime.open_gpu(0)?;
    let mut prepared = divide_by_first_prepare::<i128, _>(&session.checked_div_rem_role_backend())
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let mut quotient = [91; 5];
    let mut remainder = [91; 6];
    prepared(&[], &mut remainder, &[5, -23, i128::MIN], &mut quotient)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let expected = (
        [
            1,
            -4,
            -34_028_236_692_093_846_346_337_460_743_176_821_145,
            91,
            91,
        ],
        [0, -3, -3, 91, 91, 91],
    );
    assert_eq!((quotient, remainder), expected);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })?;
    divide_by_first::<i128>(&[], &mut remainder, &[5, -23, i128::MIN], &mut quotient)?;
    assert_eq!((quotient, remainder), expected);
    assert!(divide_by_first::<i128>(&[], &mut remainder, &[0, 7, 9], &mut quotient).is_err());
    assert_eq!((quotient, remainder), expected);
    global::clear_thread_cache()?;
    println!("MLX repeated input completed quotient {quotient:?}, remainder {remainder:?}");
    Ok(())
}

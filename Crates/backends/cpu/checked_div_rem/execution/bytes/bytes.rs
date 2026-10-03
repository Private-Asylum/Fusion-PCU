//! Isolated native-endian byte access for the sealed canonical fourteen-carrier integer executable.

use fusion_pcu::PcuCheckedIntegerDivision;
use crate::PcuCpuCheckedIntegerError;

pub(super) type Executable =
    fn(&[u8], &[u8], &mut [u8], &mut [u8], usize) -> Result<(), PcuCpuCheckedIntegerError>;

pub(super) fn prepare<T: PcuCheckedIntegerDivision>(
    left_zero: bool,
    right_zero: bool,
) -> Executable {
    match (left_zero, right_zero) {
        (false, false) => execute::<T, false, false>,
        (false, true) => execute::<T, false, true>,
        (true, false) => execute::<T, true, false>,
        (true, true) => execute::<T, true, true>,
    }
}

fn execute<T: PcuCheckedIntegerDivision, const LEFT_ZERO: bool, const RIGHT_ZERO: bool>(
    left: &[u8],
    right: &[u8],
    quotient: &mut [u8],
    remainder: &mut [u8],
    extent: usize,
) -> Result<(), PcuCpuCheckedIntegerError> {
    // Cold admission permits only fourteen sealed canonical integer representations, each with every
    // bit pattern valid. Complete original-declaration validation checks this multiplication and
    // all spans before entry. No aligned typed reference is created and no pointer escapes.
    let bytes = extent * T::HOST_SIZE;
    let left = left[..if LEFT_ZERO { T::HOST_SIZE } else { bytes }]
        .as_ptr()
        .cast::<T>();
    let right = right[..if RIGHT_ZERO { T::HOST_SIZE } else { bytes }]
        .as_ptr()
        .cast::<T>();
    let quotient = quotient[..bytes].as_mut_ptr().cast::<T>();
    let remainder = remainder[..bytes].as_mut_ptr().cast::<T>();
    let load = |index: usize| {
        // SAFETY: each selected index is zero for a frozen scalar read or below extent
        // for an indexed read. Cold counts and complete argument validation cover those
        // initialized slots. No alignment is required; admitted types have no padding.
        unsafe {
            (
                left.add(if LEFT_ZERO { 0 } else { index }).read_unaligned(),
                right
                    .add(if RIGHT_ZERO { 0 } else { index })
                    .read_unaligned(),
            )
        }
    };
    for index in 0..extent {
        let (lhs, rhs) = load(index);
        // The sealed same-width domain has exactly two failures: zero and signed MIN/-1.
        // Validate those without doing division; no other quotient or remainder range fault exists.
        lhs.pcu_div_rem_domain(rhs)
            .map_err(|kind| crate::checked_integer::fault(index, kind))?;
    }
    // Immutable inputs cannot change during synchronous execution. Whole-map preflight
    // therefore proves both publications; no later arithmetic fault can publish a prefix.
    for index in 0..extent {
        let (lhs, rhs) = load(index);
        let (q, r) = lhs
            .pcu_checked_div_rem(rhs)
            .expect("whole-map preflight succeeded");
        // SAFETY: each exclusive output borrow covers extent T slots; index is in bounds.
        // Safe argument construction makes both outputs disjoint from each other and inputs.
        unsafe {
            quotient.add(index).write_unaligned(q);
            remainder.add(index).write_unaligned(r);
        }
    }
    Ok(())
}

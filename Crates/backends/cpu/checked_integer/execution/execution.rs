//! Cold-frozen operation/layout and bounded native-endian scalar publication.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedInteger,
    PcuDispatchIntegerBinaryOp,
    PcuExecutionFaultKind,
};
use super::PcuCpuCheckedIntegerError;

pub(super) type Executable =
    fn(&[u8], &[u8], &mut [u8], usize) -> Result<(), PcuCpuCheckedIntegerError>;

pub(super) const fn prepare<T: PcuCheckedInteger>(
    op: PcuDispatchIntegerBinaryOp,
    left_broadcast: bool,
    right_broadcast: bool,
) -> Executable {
    macro_rules! layout {
        ($op:expr) => {
            match (left_broadcast, right_broadcast) {
                (false, false) => execute::<T, $op, false, false>,
                (false, true) => execute::<T, $op, false, true>,
                (true, false) => execute::<T, $op, true, false>,
                (true, true) => execute::<T, $op, true, true>,
            }
        };
    }
    match op {
        PcuDispatchIntegerBinaryOp::Add => layout!(0),
        PcuDispatchIntegerBinaryOp::Sub => layout!(1),
        PcuDispatchIntegerBinaryOp::Mul => layout!(2),
    }
}

fn execute<
    T: PcuCheckedInteger,
    const OP: u8,
    const LEFT_BROADCAST: bool,
    const RIGHT_BROADCAST: bool,
>(
    left: &[u8],
    right: &[u8],
    output: &mut [u8],
    extent: usize,
) -> Result<(), PcuCpuCheckedIntegerError> {
    // The only entry is a fully admitted, synchronous prepared call. Truncate once before
    // pointer creation to establish each complete access span independently of loop bounds.
    // Extent multiplication has already been checked for every original loaded binding.
    let left_size = if LEFT_BROADCAST {
        T::HOST_SIZE
    } else {
        extent * T::HOST_SIZE
    };
    let right_size = if RIGHT_BROADCAST {
        T::HOST_SIZE
    } else {
        extent * T::HOST_SIZE
    };
    let left = left[..left_size].as_ptr().cast::<T>();
    let right = right[..right_size].as_ptr().cast::<T>();
    let output = output[..extent * T::HOST_SIZE].as_mut_ptr().cast::<T>();
    let load = |invocation: usize| {
        // SAFETY: These spans were checked before entering either loop; broadcast accesses
        // element zero and other accesses are strictly below extent. The core trait is
        // sealed to eight padding-free integer primitives and every initialized bit pattern
        // is valid. read_unaligned creates no aligned T reference and preserves native bits.
        unsafe {
            (
                left.add(if LEFT_BROADCAST { 0 } else { invocation })
                    .read_unaligned(),
                right
                    .add(if RIGHT_BROADCAST { 0 } else { invocation })
                    .read_unaligned(),
            )
        }
    };
    for invocation in 0..extent {
        let (lhs, rhs) = load(invocation);
        evaluate::<T, OP>(lhs, rhs).map_err(|kind| super::fault(invocation, kind))?;
    }
    // Input spans may alias each other (including repeated SSA operands), but safe host
    // argument construction gives output an exclusive Rust borrow disjoint from both.
    // Inputs cannot change during this synchronous call, so exact preflight proves publication.
    for invocation in 0..extent {
        let (lhs, rhs) = load(invocation);
        let result = evaluate::<T, OP>(lhs, rhs).expect("checked preflight succeeded");
        // SAFETY: The unique output span has extent complete native T slots, invocation is
        // in bounds, and write_unaligned requires no T alignment. No pointer escapes the call.
        unsafe {
            output.add(invocation).write_unaligned(result);
        }
    }
    Ok(())
}
fn evaluate<T: PcuCheckedInteger, const OP: u8>(
    lhs: T,
    rhs: T,
) -> Result<T, PcuExecutionFaultKind> {
    match OP {
        0 => lhs.pcu_checked_add(rhs),
        1 => lhs.pcu_checked_sub(rhs),
        2 => lhs.pcu_checked_mul(rhs),
        _ => unreachable!("cold constructor selects only checked Add/Sub/Mul"),
    }
}

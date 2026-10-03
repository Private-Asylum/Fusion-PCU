//! Fourteen-width source parity includes full-width transport and range-fault rollback.

#[path = "clamped/clamped.rs"]
pub mod clamped;

#[path = "div_rem/div_rem.rs"]
pub mod div_rem;

#[path = "operands/operands.rs"]
pub mod operands;

#[cfg(feature = "tensor")]
#[path = "prefix/prefix.rs"]
pub mod prefix;

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedInteger,
    PcuExecutionFaultKind,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};

#[rustfmt::skip]
use super::support::{
    bits,
    POLICY_LOCK,
};

#[pcu(invocations = N)]
fn add<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}

#[pcu(invocations = N)]
fn sub<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}

#[pcu(invocations = N)]
fn mul<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}

#[pcu(invocations = N)]
fn reordered_add<T: PcuCheckedInteger, const N: usize>(output: &mut [T], right: &[T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N)]
fn reordered_sub<T: PcuCheckedInteger, const N: usize>(output: &mut [T], right: &[T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N)]
fn reordered_mul<T: PcuCheckedInteger, const N: usize>(output: &mut [T], right: &[T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}

trait Sample: PcuCheckedInteger {
    const MAX: Self;
    const MIN: Self;
    fn small(value: u8) -> Self;
}

macro_rules! native {
    ($($ty:ty),+ $(,)?) => {$(
        impl Sample for $ty {
            const MAX: Self = Self::MAX;
            const MIN: Self = Self::MIN;
            fn small(value: u8) -> Self {
                Self::try_from(value).unwrap()
            }
        }
    )+};
}

native!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);

macro_rules! wide {
    ($ty:ty, $width:literal) => {
        impl Sample for $ty {
            const MAX: Self = Self::MAX;
            const MIN: Self = Self::ZERO;
            fn small(value: u8) -> Self {
                let mut limbs = [0; $width];
                limbs[0] = u64::from(value);
                Self::from_limbs_le(limbs)
            }
        }
    };
}

wide!(PcuU256, 4);
wide!(PcuU512, 8);

macro_rules! signed_wide {
    ($ty:ty, $width:literal) => {
        impl Sample for $ty {
            const MAX: Self = Self::MAX;
            const MIN: Self = Self::MIN;
            fn small(value: u8) -> Self {
                let mut limbs = [0; $width];
                limbs[0] = u64::from(value);
                Self::from_limbs_le(limbs)
            }
        }
    };
}

signed_wide!(PcuI256, 4);
signed_wide!(PcuI512, 8);

fn fault(error: &global::PcuExecutionError, kind: PcuExecutionFaultKind) {
    assert!(
        matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == kind && fault.invocation_id == 2 && !fault.recovered)
    );
}

fn format<T: Sample>() {
    let sentinel = T::small(17);
    let mut output = [sentinel; 9];
    for phase in [1_u8, 2, 3] {
        let values = [2_u8, 3, 4, 5, 6, 7, 8].map(|v| v + phase);
        let left = values.map(T::small);
        let right = [T::small(2); 7];
        add::<T, 7>(&left, &right, &mut output).unwrap();
        bits(&output[..7], &values.map(|v| T::small(v + 2)));
        sub::<T, 7>(&left, &right, &mut output).unwrap();
        bits(&output[..7], &values.map(|v| T::small(v - 2)));
        mul::<T, 7>(&left, &right, &mut output).unwrap();
        bits(&output[..7], &values.map(|v| T::small(v * 2)));
        reordered_add::<T, 7>(&mut output, &right, &left).unwrap();
        bits(&output[..7], &values.map(|v| T::small(v + 2)));
        reordered_sub::<T, 7>(&mut output, &right, &left).unwrap();
        bits(&output[..7], &values.map(|v| T::small(v - 2)));
        reordered_mul::<T, 7>(&mut output, &right, &left).unwrap();
        bits(&output[..7], &values.map(|v| T::small(v * 2)));
        bits(&output[7..], &[sentinel; 2]);
    }
    // Successful maximum/minimum values require every high limb and sign bit to survive.
    let zero = [T::small(0); 7];
    let one = [T::small(1); 7];
    add::<T, 7>(&[T::MAX; 7], &zero, &mut output).unwrap();
    bits(&output[..7], &[T::MAX; 7]);
    sub::<T, 7>(&[T::MIN; 7], &zero, &mut output).unwrap();
    bits(&output[..7], &[T::MIN; 7]);
    mul::<T, 7>(&[T::MAX; 7], &one, &mut output).unwrap();
    bits(&output[..7], &[T::MAX; 7]);

    let previous = output;
    let mut bad = [T::small(2); 7];
    bad[2] = T::MAX;
    bad[6] = T::MAX;
    fault(
        &add::<T, 7>(&bad, &one, &mut output).unwrap_err(),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    bits(&output, &previous);
    fault(
        &mul::<T, 7>(&bad, &[T::small(2); 7], &mut output).unwrap_err(),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    bits(&output, &previous);
    bad[2] = T::MIN;
    bad[6] = T::MIN;
    fault(
        &sub::<T, 7>(&bad, &one, &mut output).unwrap_err(),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    bits(&output, &previous);
    mul::<T, 7>(&[T::MAX; 7], &one, &mut output).unwrap();
    bits(&output, &previous);
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    format::<u8>();
    format::<i8>();
    format::<u16>();
    format::<i16>();
    format::<u32>();
    format::<i32>();
    format::<u64>();
    format::<i64>();
    format::<u128>();
    format::<i128>();
    format::<PcuU256>();
    format::<PcuI256>();
    format::<PcuU512>();
    format::<PcuI512>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

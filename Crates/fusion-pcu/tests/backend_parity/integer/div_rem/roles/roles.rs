//! Actual-source division resource roles are independent of declared argument positions.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuI256,
    PcuI512,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuU256,
    PcuU512,
};

#[path = "portable/portable.rs"]
mod portable;

#[pcu(invocations = N)]
fn repeated<T: PcuCheckedIntegerDivision, const N: usize>(
    q: &mut [T],
    input: &[T; N],
    r: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[id]);
    r[id] = remainder;
    q[id] = quotient;
}

#[pcu(invocations = N)]
fn unread<T: PcuCheckedIntegerDivision, const N: usize>(
    _unused: &[T],
    r: &mut [T],
    input: &[T; N],
    q: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[0]);
    q[id] = quotient;
    r[id] = remainder;
}

#[pcu(invocations = N)]
fn reordered<T: PcuCheckedIntegerDivision, const N: usize>(
    right: &[T; N],
    r: &mut [T],
    left: &[T; N],
    q: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(left[id], right[id]);
    r[id] = remainder;
    q[id] = quotient;
}

#[pcu(invocations = 3)]
fn grid<T: PcuCheckedIntegerDivision, const N: usize>(
    _unused: &[T],
    r: &mut [T],
    input: &[T; N],
    q: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (quotient, remainder) = pcu::checked_div_rem(input[0], input[id]);
        r[id] = remainder;
        q[id] = quotient;
        id += stride;
    }
}

#[pcu(invocations = N)]
fn scalar_divisor<T: PcuCheckedIntegerDivision, const N: usize>(
    q: &mut [T],
    divisor: &T,
    r: &mut [T],
    input: &[T; N],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], *divisor);
    q[id] = quotient;
    r[id] = remainder;
}

#[pcu(invocations = 3)]
fn scalar_grid<T: PcuCheckedIntegerDivision, const N: usize>(input: &T, q: &mut [T], r: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (quotient, remainder) = pcu::checked_div_rem(*input, *input);
        q[id] = quotient;
        r[id] = remainder;
        id += stride;
    }
}

#[derive(Clone, Copy)]
enum Profile {
    Repeated,
    Unread,
    Reordered,
    Grid,
    ScalarDivisor,
    ScalarGrid,
}
impl Profile {
    fn call<T: Sample>(
        self,
        left: &[T; 7],
        right: &[T; 7],
        q: &mut [T],
        r: &mut [T],
    ) -> Result<(), global::PcuExecutionError> {
        match self {
            Self::Repeated => repeated::<T, 7>(q, left, r),
            Self::Unread => unread::<T, 7>(&[], r, left, q),
            Self::Reordered => reordered::<T, 7>(right, r, left, q),
            Self::Grid => grid::<T, 7>(&[], r, left, q),
            Self::ScalarDivisor => scalar_divisor::<T, 7>(q, &right[0], r, left),
            Self::ScalarGrid => scalar_grid::<T, 7>(&left[0], q, r),
        }
    }
    const fn numbers(self, left: u8, first: u8) -> (u8, u8) {
        match self {
            Self::Repeated | Self::ScalarGrid => (1, 0),
            Self::Unread => (left / first, left % first),
            Self::Grid => (first / left, first % left),
            Self::Reordered | Self::ScalarDivisor => (left / 4, left % 4),
        }
    }
    const fn zero_lane(self) -> usize {
        match self {
            Self::Unread | Self::ScalarDivisor | Self::ScalarGrid => 0,
            Self::Repeated | Self::Reordered | Self::Grid => 2,
        }
    }
}

fn assert_fault(error: &global::PcuExecutionError, lane: usize, kind: PcuExecutionFaultKind) {
    assert!(
        matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
        if fault.invocation_id == u64::try_from(lane).unwrap() && fault.kind == kind && !fault.recovered),
        "expected {kind:?} at {lane}, got {error:?}"
    );
}

fn execute_profile<T: Sample>(profile: Profile) {
    let sentinel = T::small(97);
    let mut q = [sentinel; 9];
    let mut r = [sentinel; 10];
    for phase in [0_u8, 1, 2] {
        let raw = [7_u8, 11, 15, 19, 23, 27, 31].map(|value| value + phase);
        let left = raw.map(T::small);
        let right = [T::small(4); 7];
        profile.call(&left, &right, &mut q, &mut r).unwrap();
        let cold_scores = SCORES.load(Ordering::Relaxed);
        bits(
            &q[..7],
            &raw.map(|value| T::small(profile.numbers(value, raw[0]).0)),
        );
        bits(
            &r[..7],
            &raw.map(|value| T::small(profile.numbers(value, raw[0]).1)),
        );
        bits(&q[7..], &[sentinel; 2]);
        bits(&r[7..], &[sentinel; 3]);
        let before_q = q;
        let before_r = r;
        let mut zero_left = left;
        let mut zero_right = right;
        let lane = profile.zero_lane();
        if matches!(profile, Profile::Reordered | Profile::ScalarDivisor) {
            zero_right[lane] = T::small(0);
        } else {
            zero_left[lane] = T::small(0);
        }
        assert_fault(
            &profile
                .call(&zero_left, &zero_right, &mut q, &mut r)
                .unwrap_err(),
            lane,
            PcuExecutionFaultKind::DivideByZero,
        );
        bits(&q, &before_q);
        bits(&r, &before_r);
        assert!(profile.call(&left, &right, &mut q, &mut r[..6]).is_err());
        bits(&q, &before_q);
        bits(&r, &before_r);
        if let Some((min, negative_one)) = T::SIGNED_DOMAIN {
            let mut min_left = left;
            let mut min_right = right;
            if matches!(
                profile,
                Profile::Reordered | Profile::ScalarDivisor | Profile::Grid
            ) {
                if matches!(profile, Profile::Grid) {
                    min_left[0] = min;
                    min_left[2] = negative_one;
                } else {
                    min_left[2] = min;
                    min_right[if matches!(profile, Profile::ScalarDivisor) {
                        0
                    } else {
                        2
                    }] = negative_one;
                }
                assert_fault(
                    &profile
                        .call(&min_left, &min_right, &mut q, &mut r)
                        .unwrap_err(),
                    2,
                    PcuExecutionFaultKind::SignedDivisionOverflow,
                );
                bits(&q, &before_q);
                bits(&r, &before_r);
            } else if matches!(profile, Profile::Repeated | Profile::ScalarGrid) {
                profile.call(&[min; 7], &right, &mut q, &mut r).unwrap();
                bits(&q[..7], &[T::small(1); 7]);
                bits(&r[..7], &[T::small(0); 7]);
            }
        }
        profile.call(&left, &right, &mut q, &mut r).unwrap();
        bits(&q, &before_q);
        bits(&r, &before_r);
        assert_eq!(SCORES.load(Ordering::Relaxed), cold_scores);
    }
}

fn format<T: Sample>() {
    global::clear_thread_cache().unwrap();
    for profile in [
        Profile::Repeated,
        Profile::Unread,
        Profile::Reordered,
        Profile::Grid,
        Profile::ScalarDivisor,
        Profile::ScalarGrid,
    ] {
        execute_profile::<T>(profile);
    }
}

pub fn verify(backend: global::PcuBackendChoice) {
    verify_requirement(backend, PcuReproducibility::Unspecified);
}

/// Required native Portable opt-in gate; neutral eligibility is insufficient.
pub fn verify_portable(backend: global::PcuBackendChoice) {
    verify_requirement(backend, PcuReproducibility::PortableV1);
}

fn verify_requirement(backend: global::PcuBackendChoice, reproducibility: PcuReproducibility) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    let policy = global::PcuExecutionPolicy {
                        backend,
                        numerical_mode,
                        float_underflow,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            reproducibility,
                        },
                        score_invocation: Some(score),
                        ..Default::default()
                    };
                    global::configure(policy).unwrap();
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
                    if reproducibility == PcuReproducibility::PortableV1 {
                        // Genuine function-local deterministic flags must work
                        // even when the global environment does not request it.
                        let mut inherited = policy;
                        inherited.numerical_options.reproducibility =
                            PcuReproducibility::Unspecified;
                        global::configure(inherited).unwrap();
                        portable::verify();
                    }
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

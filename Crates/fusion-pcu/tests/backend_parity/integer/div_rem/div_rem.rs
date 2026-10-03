//! Exact native-width quotient/remainder, two-output rollback, signed domains and retry.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionFaultKind,
    PcuCheckedIntegerDivision,
};
#[rustfmt::skip]
use super::super::support::{
    bits,
    POLICY_LOCK,
};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

trait Sample: PcuCheckedIntegerDivision {
    const SIGNED_DOMAIN: Option<(Self, Self)>;
    fn small(value: u8) -> Self;
}

macro_rules! samples {
    ($($unsigned:ty, $signed:ty);+ $(;)?) => {$(
        impl Sample for $unsigned {
            const SIGNED_DOMAIN: Option<(Self, Self)> = None;
            fn small(value: u8) -> Self { Self::from(value) }
        }
        impl Sample for $signed {
            const SIGNED_DOMAIN: Option<(Self, Self)> = Some((Self::MIN, -1));
            fn small(value: u8) -> Self { Self::try_from(value).unwrap() }
        }
    )+};
}
samples!(u8, i8; u16, i16; u32, i32; u64, i64);

#[path = "wide/wide.rs"]
mod wide;
pub use wide::verify as verify_fourteen;

#[path = "roles/roles.rs"]
mod roles;
pub use roles::verify as verify_roles;
pub use roles::verify_portable;

fn fault(error: &global::PcuExecutionError, kind: PcuExecutionFaultKind) {
    assert!(
        matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
        if fault.invocation_id == 2 && fault.kind == kind && !fault.recovered),
        "expected {kind:?} at logical invocation2, received {error:?}"
    );
}

fn execute<T: Sample>(
    call: impl Fn(&[T; 7], &[T; 7], &mut [T], &mut [T]) -> Result<(), global::PcuExecutionError>,
) {
    let sentinel = T::small(97);
    let mut quotient = [sentinel; 9];
    let mut remainder = [sentinel; 10];
    for phase in [0_u8, 1, 2] {
        let raw = [7_u8, 11, 15, 19, 23, 27, 31].map(|value| value + phase);
        let lhs = raw.map(T::small);
        let rhs = [T::small(4); 7];
        call(&lhs, &rhs, &mut quotient, &mut remainder).unwrap();
        let cold_scores = SCORES.load(Ordering::Relaxed);
        bits(&quotient[..7], &raw.map(|value| T::small(value / 4)));
        bits(&remainder[..7], &raw.map(|value| T::small(value % 4)));
        bits(&quotient[7..], &[sentinel; 2]);
        bits(&remainder[7..], &[sentinel; 3]);

        let prior_q = quotient;
        let prior_r = remainder;
        let mut zero_divisor = rhs;
        zero_divisor[2] = T::small(0);
        zero_divisor[6] = T::small(0);
        fault(
            &call(&lhs, &zero_divisor, &mut quotient, &mut remainder).unwrap_err(),
            PcuExecutionFaultKind::DivideByZero,
        );
        bits(&quotient, &prior_q);
        bits(&remainder, &prior_r);
        if let Some((min, negative_one)) = T::SIGNED_DOMAIN {
            let mut invalid_lhs = lhs;
            let mut invalid_rhs = rhs;
            invalid_lhs[2] = min;
            invalid_lhs[6] = min;
            invalid_rhs[2] = negative_one;
            invalid_rhs[6] = negative_one;
            fault(
                &call(&invalid_lhs, &invalid_rhs, &mut quotient, &mut remainder).unwrap_err(),
                PcuExecutionFaultKind::SignedDivisionOverflow,
            );
            bits(&quotient, &prior_q);
            bits(&remainder, &prior_r);
            // MIN / 1 is exact; rejecting MIN itself would be a false range fault.
            call(&[min; 7], &[T::small(1); 7], &mut quotient, &mut remainder).unwrap();
            bits(&quotient[..7], &[min; 7]);
            bits(&remainder[..7], &[T::small(0); 7]);
        }
        call(&lhs, &rhs, &mut quotient, &mut remainder).unwrap();
        bits(&quotient, &prior_q);
        bits(&remainder, &prior_r);
        assert_eq!(SCORES.load(Ordering::Relaxed), cold_scores);
    }
}

macro_rules! profiles {
    ($($module:ident: $ty:ty),+ $(,)?) => {$(
        mod $module {
            use super::*;
            #[pcu(invocations = N)]
            fn direct<const N: usize>(lhs: &[$ty; N], rhs: &[$ty; N], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
                quotient[id] = q;
                remainder[id] = r;
            }
            #[pcu(invocations = 3)]
            fn grid<const N: usize>(lhs: &[$ty; N], rhs: &[$ty; N], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let mut id = pcu::context::global_invocation_id();
                let stride = pcu::context::invocation_count();
                while id < N {
                    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
                    quotient[id] = q;
                    remainder[id] = r;
                    id += stride;
                }
            }
            pub fn verify() {
                global::clear_thread_cache().unwrap();
                execute::<$ty>(direct::<7>);
                execute::<$ty>(grid::<7>);
            }
        }
    )+};
}
profiles!(unsigned8: u8, signed8: i8, unsigned16: u16, signed16: i16,
    unsigned32: u32, signed32: i32, unsigned64: u64, signed64: i64);

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [
        fusion_pcu::PcuNumericalMode::Boundary,
        fusion_pcu::PcuNumericalMode::Strict,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend,
            numerical_mode,
            score_invocation: Some(score),
            ..Default::default()
        })
        .unwrap();
        unsigned8::verify();
        signed8::verify();
        unsigned16::verify();
        signed16::verify();
        unsigned32::verify();
        signed32::verify();
        unsigned64::verify();
        signed64::verify();
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

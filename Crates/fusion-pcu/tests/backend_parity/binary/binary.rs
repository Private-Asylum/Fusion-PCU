//! Exact dyadic inputs give a backend-independent bit oracle without native float rounding.
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;

#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
};

use super::source;

#[cfg(feature = "tensor")]
#[path = "prefix/prefix.rs"]
pub mod prefix;
#[rustfmt::skip]
use super::support::{
    bits,
    POLICY_LOCK,
};

static SCORES: AtomicUsize = AtomicUsize::new(0);

trait Sample: PcuCheckedFloat {
    fn finite(value: f32) -> Self;
}

macro_rules! low_sample {
    ($($ty:ty),+ $(,)?) => {$(
        impl Sample for $ty {
            fn finite(value: f32) -> Self {
                Self::pcu_checked_from_f32(value).unwrap()
            }
        }
    )+};
}

low_sample!(PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits);

impl Sample for f32 {
    fn finite(value: f32) -> Self {
        value
    }
}

impl Sample for f64 {
    fn finite(value: f32) -> Self {
        Self::from(value)
    }
}

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}

fn format<T: Sample>() {
    let sentinel = T::finite(16.0);
    let mut output = [sentinel; 9];
    for multiplier in [1.0, 2.0, 3.0] {
        let raw_left = [0.5, 1.0, 2.0, 4.0, 1.0, 0.5, 2.0].map(|v| v * multiplier);
        let raw_right = [2.0; 7];
        let left = raw_left.map(T::finite);
        let right = raw_right.map(T::finite);
        // All expected dyadic values are exactly representable in every tested format.
        source::add::<T, 7>(&left, &right, &mut output).unwrap();
        bits(&output[..7], &raw_left.map(|v| T::finite(v + 2.0)));
        source::sub::<T, 7>(&left, &right, &mut output).unwrap();
        bits(&output[..7], &raw_left.map(|v| T::finite(v - 2.0)));
        source::mul::<T, 7>(&left, &right, &mut output).unwrap();
        bits(&output[..7], &raw_left.map(|v| T::finite(v * 2.0)));
        source::div::<T, 7>(&left, &right, &mut output).unwrap();
        bits(&output[..7], &raw_left.map(|v| T::finite(v / 2.0)));
        bits(&output[7..], &[sentinel; 2]);

        let before = output;
        let mut bad_right = right;
        bad_right[2] = T::finite(0.0);
        bad_right[6] = T::finite(0.0);
        let error = source::div::<T, 7>(&left, &bad_right, &mut output).unwrap_err();
        assert!(
            matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
            if fault.kind == PcuExecutionFaultKind::DivideByZero
                && fault.invocation_id == 2 && !fault.recovered)
        );
        bits(&output, &before);
        source::div::<T, 7>(&left, &right, &mut output).unwrap();
        bits(&output, &before);
    }
}

fn all_formats() {
    format::<PcuF16Bits>();
    format::<PcuBf16Bits>();
    format::<PcuF8E4M3FnBits>();
    format::<PcuF8E5M2Bits>();
    format::<f32>();
    format::<f64>();
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for float_underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            global::configure(global::PcuExecutionPolicy {
                backend,
                numerical_mode,
                float_underflow,
                score_invocation: Some(score),
                ..Default::default()
            })
            .unwrap();
            global::clear_thread_cache().unwrap();
            all_formats();
            let cold = SCORES.load(Ordering::Relaxed);
            all_formats();
            assert_eq!(SCORES.load(Ordering::Relaxed), cold);
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

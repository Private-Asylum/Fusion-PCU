//! An unread declaration is metadata, never a storage, affinity or staging request.
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;

#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuCompoundArithmeticPolicy,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
};

#[rustfmt::skip]
use super::support::{
    bits,
    POLICY_LOCK,
};

#[path = "source/source.rs"]
mod source;

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}

fn format<T: PcuCheckedFloat>(finite: impl Fn(f32) -> T + Copy) {
    let sentinel = finite(16.0);
    let empty: [T; 0] = [];
    let mut output = [sentinel; 9];
    for scale in [1.0, 2.0, 0.5] {
        let raw = [1.0, 0.5, 2.0, 4.0, 1.0, 2.0, 0.5].map(|v| v * scale);
        let input = raw.map(finite);
        source::one::<T, 7>(&input, &mut output).unwrap();
        bits(&output[..7], &raw.map(|v| finite(v + v)));
        source::repeated::<T, 7>(&input, &empty, &mut output).unwrap();
        bits(&output[..7], &raw.map(|v| finite(v * v)));
        source::reordered::<T, 7>(&mut output, &empty, &input).unwrap();
        bits(&output[..7], &[finite(0.0); 7]);
        source::independent::<T, 7>(&input, &mut output).unwrap();
        bits(&output[..7], &raw.map(|v| finite(v / raw[0])));
        source::grid::<T, 7>(&empty, &mut output, &input).unwrap();
        bits(&output[..7], &[finite(1.0); 7]);
        bits(&output[7..], &[sentinel; 2]);

        // Repeated SSA loads are still two mathematical operands. A real zero
        // divisor must fault even when the other declaration has no storage.
        let before = output;
        let mut zero = input;
        zero[2] = finite(0.0);
        zero[6] = finite(0.0);
        let error = source::grid::<T, 7>(&empty, &mut output, &zero).unwrap_err();
        let fault = error.arithmetic_fault().expect("division fault");
        assert_eq!(fault.kind, PcuExecutionFaultKind::DivideByZero);
        assert_eq!(fault.invocation_id, 2);
        assert!(!fault.recovered);
        bits(&output, &before);
        source::grid::<T, 7>(&empty, &mut output, &input).unwrap();
        bits(&output, &before);
    }
}

fn all_formats() {
    format::<PcuF16Bits>(|v| PcuF16Bits::pcu_checked_from_f32(v).unwrap());
    format::<PcuBf16Bits>(|v| PcuBf16Bits::pcu_checked_from_f32(v).unwrap());
    format::<PcuF8E4M3FnBits>(|v| PcuF8E4M3FnBits::pcu_checked_from_f32(v).unwrap());
    format::<PcuF8E5M2Bits>(|v| PcuF8E5M2Bits::pcu_checked_from_f32(v).unwrap());
    format::<f32>(|v| v);
    format::<f64>(f64::from);
}

pub fn verify(backend: global::PcuBackendChoice) {
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
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        global::configure(global::PcuExecutionPolicy {
                            backend,
                            numerical_mode,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                ..Default::default()
                            },
                            float_underflow,
                            range_policy,
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
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

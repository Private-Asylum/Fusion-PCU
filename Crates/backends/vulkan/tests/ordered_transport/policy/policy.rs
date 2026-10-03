//! Transport retains complete requested numerical tuples even when the payload operation is inert.
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
};
pub fn each(mut run: impl FnMut(PcuImplementationRequirements)) {
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
                        run(PcuImplementationRequirements {
                            numerical_mode,
                            float_underflow,
                            range_policy,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                ..Default::default()
                            },
                        });
                    }
                }
            }
        }
    }
}
pub fn configure(
    requirements: PcuImplementationRequirements,
    score: Option<fn(&global::PcuInvocationCandidate<'_>) -> i128>,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        numerical_mode: requirements.numerical_mode,
        numerical_options: requirements.numerical_options,
        float_underflow: requirements.float_underflow,
        range_policy: requirements.range_policy,
        score_invocation: score,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
#[macro_export]
macro_rules! carriers {
    ($run:ident $(,$arg:expr)*) => {{
        $run::<u8>($($arg),*); $run::<i8>($($arg),*);
        $run::<u16>($($arg),*); $run::<i16>($($arg),*);
        $run::<u32>($($arg),*); $run::<i32>($($arg),*);
        $run::<u64>($($arg),*); $run::<i64>($($arg),*);
        $run::<u128>($($arg),*); $run::<i128>($($arg),*);
        $run::<pcu_facade::PcuU256>($($arg),*); $run::<pcu_facade::PcuI256>($($arg),*);
        $run::<pcu_facade::PcuU512>($($arg),*); $run::<pcu_facade::PcuI512>($($arg),*);
        $run::<pcu_facade::PcuF16Bits>($($arg),*); $run::<pcu_facade::PcuBf16Bits>($($arg),*);
        $run::<pcu_facade::PcuF8E4M3FnBits>($($arg),*); $run::<pcu_facade::PcuF8E5M2Bits>($($arg),*);
        $run::<f32>($($arg),*); $run::<f64>($($arg),*);
        $run::<pcu_facade::PcuF128Bits>($($arg),*); $run::<pcu_facade::PcuF256Bits>($($arg),*);
    }};
}

//! All fourteen exact division widths share the same genuine annotated joint-output source.
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
    PcuU256,
    PcuU512,
};

samples!(u128, i128);

macro_rules! wide_samples {
    ($unsigned:ty, $signed:ty, $limbs:expr) => {
        impl Sample for $unsigned {
            const SIGNED_DOMAIN: Option<(Self, Self)> = None;
            fn small(value: u8) -> Self {
                <Self as super::super::Sample>::small(value)
            }
        }
        impl Sample for $signed {
            const SIGNED_DOMAIN: Option<(Self, Self)> =
                Some((Self::MIN, Self::from_limbs_le($limbs)));
            fn small(value: u8) -> Self {
                <Self as super::super::Sample>::small(value)
            }
        }
    };
}
wide_samples!(PcuU256, PcuI256, [u64::MAX; 4]);
wide_samples!(PcuU512, PcuI512, [u64::MAX; 8]);

#[pcu(invocations = N)]
fn direct<T: PcuCheckedIntegerDivision, const N: usize>(
    lhs: &[T; N],
    rhs: &[T; N],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
    quotient[id] = q;
    remainder[id] = r;
}

#[pcu(invocations = 3)]
fn grid<T: PcuCheckedIntegerDivision, const N: usize>(
    lhs: &[T; N],
    rhs: &[T; N],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
        quotient[id] = q;
        remainder[id] = r;
        id += stride;
    }
}

fn format<T: Sample>() {
    global::clear_thread_cache().unwrap();
    execute::<T>(direct::<T, 7>);
    execute::<T>(grid::<T, 7>);
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
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    global::configure(global::PcuExecutionPolicy {
                        backend,
                        numerical_mode,
                        float_underflow,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            ..Default::default()
                        },
                        score_invocation: Some(score),
                        ..Default::default()
                    })
                    .unwrap();
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
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

//! Exact selection profiles preserve global and function-local Portable requests.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    describe_portable_v1_unary_map,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
};

#[pcu(invocations = N, flag(deterministic))]
fn direct_neg<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = N, flag(deterministic))]
fn direct_relu<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(invocations = 3, flag(deterministic))]
fn grid_neg<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}

#[pcu(invocations = 3, flag(deterministic))]
fn grid_relu<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}

#[pcu(invocations = 3, flag(deterministic))]
fn broadcast_relu<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = pcu::relu(input[0]);
        id += stride;
    }
}

fn broadcast<T: Sample>(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    let sentinel = T::from_raw(T::NORMAL);
    let mut output = [sentinel; 9];
    for raw in [0, T::SIGN, 1, T::SIGN | 1, T::NORMAL] {
        output.fill(sentinel);
        let input = [T::from_raw(raw)];
        let selected = if raw & T::SIGN == 0 { raw } else { 0 };
        let result = broadcast_relu::<T, 7>(&input, &mut output);
        let exceptional =
            selected == 1 && underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult;
        if exceptional {
            let error = result.unwrap_err();
            assert!(
                matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
                if fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow
                    && fault.invocation_id == 0 && fault.recovered == (range == PcuRangePolicy::Clamp))
            );
        } else {
            result.unwrap();
        }
        if !exceptional || range == PcuRangePolicy::Clamp {
            bits(&output[..7], &[T::from_raw(selected); 7]);
        } else {
            bits(&output[..7], &[sentinel; 7]);
        }
        bits(&output[7..], &[sentinel; 2]);
    }
    output.fill(sentinel);
    let error = broadcast_relu::<T, 7>(&[T::from_raw(T::NONFINITE)], &mut output).unwrap_err();
    assert!(
        matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand && fault.invocation_id == 0 && !fault.recovered)
    );
    bits(&output, &[sentinel; 9]);
    broadcast_relu::<T, 7>(&[T::from_raw(T::NORMAL)], &mut output).unwrap();
    bits(&output[..7], &[sentinel; 7]);
}

fn format<T: Sample>(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    exercise(underflow, range, direct_neg::<T, 7>, direct_relu::<T, 7>);
    exercise(underflow, range, grid_neg::<T, 7>, grid_relu::<T, 7>);
    broadcast::<T>(underflow, range);
}

fn all_formats(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    format::<PcuF16Bits>(underflow, range);
    format::<PcuBf16Bits>(underflow, range);
    format::<PcuF8E4M3FnBits>(underflow, range);
    format::<PcuF8E5M2Bits>(underflow, range);
    format::<f32>(underflow, range);
    format::<f64>(underflow, range);
}

pub fn verify(backend: global::PcuBackendChoice) {
    // The normal source inherits Portable globally; this preserves the existing
    // independent bit/fault oracle without cloning provider arithmetic.
    verify_options(
        backend,
        PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..Default::default()
        },
    );
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
                    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        global::configure(global::PcuExecutionPolicy {
                            backend,
                            numerical_mode,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                reproducibility: PcuReproducibility::Unspecified,
                            },
                            float_underflow,
                            range_policy,
                            score_invocation: Some(score),
                            ..Default::default()
                        })
                        .unwrap();
                        global::clear_thread_cache().unwrap();
                        // Each source function itself requests Portable even though
                        // global defaults remain Unspecified; other axes inherit.
                        all_formats(float_underflow, range_policy);
                        let cold = SCORES.load(Ordering::Relaxed);
                        all_formats(float_underflow, range_policy);
                        assert_eq!(SCORES.load(Ordering::Relaxed), cold);
                    }
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

fn source_metadata<T: PcuCheckedFloat>() {
    let bindings = direct_neg_bindings::<T>();
    let lowered = direct_neg_ir::<T, 7>(&bindings).unwrap();
    let descriptor = describe_portable_v1_unary_map(&lowered.ir()).unwrap();
    assert_eq!(
        descriptor.requirements.numerical_options.reproducibility,
        PcuReproducibility::PortableV1
    );
    assert_eq!(descriptor.scalar, T::TYPE);
    let bindings = broadcast_relu_bindings::<T>();
    broadcast_relu_ir::<T, 7>(&bindings)
        .unwrap()
        .with_ir(|kernel| {
            let descriptor = describe_portable_v1_unary_map(kernel).unwrap();
            assert_eq!(descriptor.submitted_invocations, 3);
            assert_eq!(descriptor.logical_extent, 7);
            assert_eq!(descriptor.input_extent(), 1);
        });
}

#[test]
fn annotated_portable_unary_metadata_remains_cold_structural_evidence() {
    source_metadata::<PcuF16Bits>();
    source_metadata::<PcuBf16Bits>();
    source_metadata::<PcuF8E4M3FnBits>();
    source_metadata::<PcuF8E5M2Bits>();
    source_metadata::<f32>();
    source_metadata::<f64>();
}

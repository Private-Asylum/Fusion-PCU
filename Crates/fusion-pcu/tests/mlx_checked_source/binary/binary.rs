//! Same-name source binary dispatch over host, borrowed owners and immutable outputs.

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuCompoundArithmeticPolicy,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(invocations = N)]
fn add<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N)]
fn sub<T: PcuCheckedFloat, const N: usize>(output: &mut [T], right: &[T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N)]
fn repeated<T: PcuCheckedFloat, const N: usize>(left: &[T], unused: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * left[id];
}
#[pcu(invocations = 1)]
fn div<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = N, flag(clamp_range))]
fn clamp<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}

fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}
fn read<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T; 7], sentinel: T) {
    let mut stack = [sentinel; 9];
    owner.read_into(&mut stack).unwrap();
    bits(&stack[..7], expected);
    bits(&stack[7..], &[sentinel; 2]);
}

fn format<T: PcuCheckedFloat>(finite: impl Fn(f32) -> T + Copy, maximum: T) {
    // All successful values/results are exactly representable in each low format.
    let left = [1.0, 2.0, 3.0, 4.0, -1.0].map(finite);
    let right = [2.0; 5].map(finite);
    let sentinel = finite(16.0);
    let left_owner = retain(&left).unwrap();
    let right_owner = retain(&right).unwrap();
    let mut output = retain(&[sentinel; 7]).unwrap();
    let mut host = [sentinel; 7];
    add::<T, 5>(&left, &right, &mut host).unwrap();
    bits(&host, &[3.0, 4.0, 5.0, 6.0, 1.0, 16.0, 16.0].map(finite));
    add::<T, 5>(&left_owner, &right, &mut output).unwrap();
    read(
        &output,
        &[3.0, 4.0, 5.0, 6.0, 1.0, 16.0, 16.0].map(finite),
        sentinel,
    );
    add::<T, 5>(&left, &right_owner, &mut output).unwrap();
    sub::<T, 5>(&mut output, &right_owner, &left_owner).unwrap();
    read(
        &output,
        &[-1.0, 0.0, 1.0, 2.0, -3.0, 16.0, 16.0].map(finite),
        sentinel,
    );
    repeated::<T, 5>(&left, &[], &mut host).unwrap();
    bits(&host, &[1.0, 4.0, 9.0, 16.0, 1.0, 16.0, 16.0].map(finite));
    repeated::<T, 5>(&left_owner, &[], &mut output).unwrap();
    read(
        &output,
        &[1.0, 4.0, 9.0, 16.0, 1.0, 16.0, 16.0].map(finite),
        sentinel,
    );
    div::<T, 5>(&left_owner, &right_owner, &mut host).unwrap();
    bits(&host, &[0.5, 1.0, 1.5, 2.0, -0.5, 16.0, 16.0].map(finite));
    div::<T, 5>(&left_owner, &right_owner, &mut output).unwrap();
    let before = [0.5, 1.0, 1.5, 2.0, -0.5, 16.0, 16.0].map(finite);
    read(&output, &before, sentinel);
    let zero = [finite(0.0); 5];
    let error = div::<T, 5>(&left_owner, &zero, &mut output).unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::DivideByZero);
    assert_eq!(fault.invocation_id, 0);
    assert!(!fault.recovered);
    read(&output, &before, sentinel); // Fatal private work retains the old immutable owner.
    let mut exceptional = left;
    exceptional[2] = maximum;
    let exceptional_owner = retain(&exceptional).unwrap();
    let sibling = retain(&output).unwrap();
    let error = clamp::<T, 5>(&exceptional_owner, &right_owner, &mut output).unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(fault.invocation_id, 2);
    assert!(fault.recovered);
    let mut recovered = [2.0, 4.0, 0.0, 8.0, -2.0, 16.0, 16.0].map(finite);
    recovered[2] = maximum;
    read(&output, &recovered, sentinel);
    read(&sibling, &before, sentinel);
    global::clear_thread_cache().unwrap();
    add::<T, 5>(&left_owner, &right_owner, &mut output).unwrap();
    drop(left_owner);
    drop(right_owner);
    read(
        &output,
        &[3.0, 4.0, 5.0, 6.0, 1.0, 16.0, 16.0].map(finite),
        sentinel,
    );
    read(&sibling, &before, sentinel);
}

pub fn verify() {
    let _guard = super::POLICY_LOCK.lock().unwrap();
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
                    global::configure(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Mlx,
                        numerical_mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            ..Default::default()
                        },
                        float_underflow,
                        ..Default::default()
                    })
                    .unwrap();
                    format(
                        |value| PcuF16Bits::pcu_checked_from_f32(value).unwrap(),
                        PcuF16Bits::from_bits(0x7bff),
                    );
                    format(
                        |value| PcuBf16Bits::pcu_checked_from_f32(value).unwrap(),
                        PcuBf16Bits::from_bits(0x7f7f),
                    );
                    format(
                        |value| PcuF8E4M3FnBits::pcu_checked_from_f32(value).unwrap(),
                        PcuF8E4M3FnBits::from_bits(0x7e),
                    );
                    format(
                        |value| PcuF8E5M2Bits::pcu_checked_from_f32(value).unwrap(),
                        PcuF8E5M2Bits::from_bits(0x7b),
                    );
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

//! One owned-source contract for four low floating formats on every active provider.
//!
//! Expected dyadic results are exact in all four encodings. Exceptional fixtures
//! use representation constants; the scalar implementation is not its own oracle.
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
use super::contract::POLICY_LOCK;

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn add<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}

#[pcu]
fn sub<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sub(left, right)
}

#[pcu]
fn mul<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(left, right)
}

#[pcu]
fn div<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::div(left, right)
}

#[pcu]
fn relu<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}

#[pcu]
fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn unused_checked<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _checked = pcu::relu(input)?;
    pcu::identity(input)
}

#[pcu(flag(reject_subnormal_result))]
fn local_reject<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}

#[pcu(flag(allow_gradual_underflow))]
fn local_allow<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}

trait Sample: PcuCheckedFloat {
    const TINY: Self;
    const MAX: Self;
    const INVALID: Self;
    fn finite(value: f32) -> Self;
}

macro_rules! samples {
    ($($ty:ty, $max:expr, $invalid:expr;)+) => {$(
        impl Sample for $ty {
            const TINY: Self = Self::from_bits(1);
            const MAX: Self = Self::from_bits($max);
            const INVALID: Self = Self::from_bits($invalid);
            fn finite(value: f32) -> Self {
                Self::pcu_checked_from_f32(value).unwrap()
            }
        }
    )+};
}
samples! {
    PcuF16Bits, 0x7bff, 0x7c00;
    PcuBf16Bits, 0x7f7f, 0x7f80;
    PcuF8E4M3FnBits, 0x7e, 0x7f;
    PcuF8E5M2Bits, 0x7b, 0x7c;
}

fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}

fn read<T: Sample>(owner: &PcuTensor<T>, expected: &[T; 7]) {
    assert_eq!(owner.shape(), &[7]);
    let sentinel = T::finite(16.0);
    let mut stack = [sentinel; 9];
    owner.read_into(&mut stack).unwrap();
    bits(&stack[..7], expected);
    bits(&stack[7..], &[sentinel; 2]);
}

fn fault<T: PcuScalar>(
    result: Result<PcuTensor<T>, PcuExecutionError>,
    kind: PcuExecutionFaultKind,
) {
    let error = result.unwrap_err();
    let fault = error
        .arithmetic_fault()
        .unwrap_or_else(|| panic!("expected fatal {kind:?} at lane2, received {error:?}"));
    assert_eq!(fault.kind, kind);
    assert_eq!(fault.invocation_id, 2);
    assert!(!fault.recovered);
}

fn exceptions<T: Sample>(sibling: &PcuTensor<T>, policy: PcuFloatUnderflowPolicy) {
    let one = [T::finite(1.0); 7];
    let mut input = one;
    input[2] = T::finite(0.0);
    input[6] = T::finite(0.0);
    let zero_divisors = retain(&input).unwrap();
    fault(
        div::<T>(&one, &zero_divisors),
        PcuExecutionFaultKind::DivideByZero,
    );
    read(&zero_divisors, &input);
    input[2] = T::INVALID;
    input[6] = T::INVALID;
    let invalid = retain(&input).unwrap();
    fault(
        relu::<T>(&invalid),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    fault(
        unused_checked::<T>(&invalid),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    read(&invalid, &input);

    input[2] = T::MAX;
    input[6] = T::MAX;
    let maximum = retain(&input).unwrap();
    fault(
        mul::<T>(&maximum, &[T::finite(2.0); 7]),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    read(&maximum, &input);

    input[2] = T::TINY;
    input[6] = T::TINY;
    fault(
        local_reject::<T>(&input),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    read(&local_allow::<T>(&input).unwrap(), &input);
    let tiny = retain(&input).unwrap();
    if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
        fault(
            mul::<T>(&tiny, &one),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
    } else {
        // IEEE 754-2019 clause7.5: exact subnormal multiplication is not underflow.
        read(&mul::<T>(&tiny, &one).unwrap(), &input);
    }
    let half = [T::finite(0.5); 7];
    if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
        // Half the minimum subnormal rounds to +0 under nearest, ties-to-even.
        let mut expected = half;
        expected[2] = T::finite(0.0);
        expected[6] = T::finite(0.0);
        read(&mul::<T>(&tiny, &half).unwrap(), &expected);
    } else {
        fault(
            mul::<T>(&tiny, &half),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
    }
    read(&tiny, &input);
    read(sibling, &one);
    read(&relu::<T>(&one).unwrap(), &one);
}

fn format<T: Sample>(policy: PcuFloatUnderflowPolicy) {
    global::clear_thread_cache().unwrap();
    for multiplier in [1.0, 2.0, 3.0] {
        let raw = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0].map(|v| v * multiplier);
        let left = raw.map(T::finite);
        let right = [T::finite(2.0); 7];
        let a = retain(&left).unwrap();
        let b = retain(&right).unwrap();
        let sum = add::<T>(&a, &b).unwrap();
        let restored = sub::<T>(&sum, &right).unwrap();
        let product = mul::<T>(&left, &b).unwrap();
        let quotient = div::<T>(&product, &b).unwrap();
        let activated = relu::<T>(&product).unwrap();
        let unchanged = unused_checked::<T>(&a).unwrap();
        read(&a, &left);
        read(&b, &right);
        read(&sum, &raw.map(|v| T::finite(v + 2.0)));
        read(&restored, &left);
        read(&product, &raw.map(|v| T::finite(v * 2.0)));
        read(&quotient, &left);
        read(&unchanged, &left);
        drop(a);
        drop(sum);
        drop(restored);
        drop(product);
        drop(quotient);
        drop(unchanged);
        let moved = consume(activated).unwrap();
        read(&moved, &raw.map(|v| T::finite((v * 2.0).max(0.0))));
    }
    let sibling = retain(&[T::finite(1.0); 7]).unwrap();
    exceptions(&sibling, policy);
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
                    global::configure(global::PcuExecutionPolicy {
                        backend,
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
                    format::<PcuF16Bits>(float_underflow);
                    format::<PcuBf16Bits>(float_underflow);
                    format::<PcuF8E4M3FnBits>(float_underflow);
                    format::<PcuF8E5M2Bits>(float_underflow);
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

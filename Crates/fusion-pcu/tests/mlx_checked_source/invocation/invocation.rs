//! Source borrows retain immutable MLX arrays; exclusive outputs publish terminal replacements.
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
fn neg<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = N)]
fn permuted<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = N, flag(clamp_range), flag(reject_subnormal_result))]
fn clamp<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}
fn read<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T; 5], sentinel: T) {
    let mut stack = [sentinel; 7];
    owner.read_into(&mut stack).unwrap();
    bits(&stack[..5], expected);
    bits(&stack[5..], &[sentinel; 2]);
}
fn read_extended<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T; 7], sentinel: T) {
    assert_eq!(owner.shape(), &[7]);
    let mut stack = [sentinel; 9];
    owner.read_into(&mut stack).unwrap();
    bits(&stack[..7], expected);
    bits(&stack[7..], &[sentinel; 2]);
}

fn extended_output<T: PcuCheckedFloat>(finite: impl Fn(f32) -> T + Copy) {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0].map(finite);
    let sentinel = finite(16.0);
    let expected = [-1.0, -2.0, -3.0, -4.0, -5.0, 16.0, 16.0].map(finite);
    let source = retain(&input).unwrap();
    let mut output = retain(&[sentinel; 7]).unwrap();
    for _ in 0..3 {
        // Same source specialization as the exact-N case; actual output extent
        // selects a separate cold prefix plan. Warm calls preserve both tail elements.
        neg::<T, 5>(&input, &mut output).unwrap();
        read_extended(&output, &expected, sentinel);
        neg::<T, 5>(&source, &mut output).unwrap();
        permuted::<T, 5>(&mut output, &source).unwrap();
        read_extended(&output, &expected, sentinel);
        read(&source, &input, sentinel);
    }
    let mut short = retain(&[sentinel; 4]).unwrap();
    assert!(neg::<T, 5>(&source, &mut short).is_err());
    let mut stack = [sentinel; 6];
    short.read_into(&mut stack).unwrap();
    bits(&stack, &[sentinel; 6]);
    drop(short);
    drop(source);
    global::clear_thread_cache().unwrap();
    read_extended(&output, &expected, sentinel);
    // Only the escaped output retains this session now. Clearing compiled source
    // entries must not split subsequent host-created values into a foreign context.
    let source = retain(&input).unwrap();
    neg::<T, 5>(&source, &mut output).unwrap();
    read_extended(&output, &expected, sentinel);
}
fn fatal(result: &Result<(), PcuExecutionError>) {
    assert!(
        matches!(result, Err(PcuExecutionError::ArithmeticFault(fault))
        if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand
        && fault.invocation_id == 2 && !fault.recovered)
    );
}
macro_rules! format {
    ($ty:ty, $sign:expr, $normal:expr, $invalid:expr) => {{
        global::clear_thread_cache().unwrap();
        for phase in [0, 1, 2] {
            let raw = [$normal + phase, $sign | $normal, 0, $sign, $normal + 3];
            let input = raw.map(<$ty>::from_bits);
            let expected = raw.map(|bits| <$ty>::from_bits(bits ^ $sign));
            let sentinel = <$ty>::from_bits($normal);
            let source = retain(&input).unwrap();
            let sibling = retain(&input).unwrap();
            let mut output = retain(&[sentinel; 5]).unwrap();
            let mut host = [sentinel; 7];
            neg::<$ty, 5>(&source, &mut host).unwrap();
            bits(&host[..5], &expected);
            bits(&host[5..], &[sentinel; 2]);
            neg::<$ty, 5>(&input, &mut output).unwrap();
            read(&output, &expected, sentinel);
            neg::<$ty, 5>(&source, &mut output).unwrap();
            permuted::<$ty, 5>(&mut output, &source).unwrap();
            read(&output, &expected, sentinel);
            read(&source, &input, sentinel);
            read(&sibling, &input, sentinel);
            let mut invalid = input;
            invalid[2] = <$ty>::from_bits($invalid);
            let invalid_owner = retain(&invalid).unwrap();
            fatal(&neg::<$ty, 5>(&invalid_owner, &mut output));
            read(&output, &expected, sentinel);
            read(&invalid_owner, &invalid, sentinel);
            fatal(&neg::<$ty, 5>(&invalid, &mut output));
            read(&output, &expected, sentinel);
            fatal(&neg::<$ty, 5>(&invalid_owner, &mut host));
            bits(&host[..5], &expected);
            bits(&host[5..], &[sentinel; 2]);
            let mut tiny = input;
            tiny[2] = <$ty>::from_bits(1);
            let tiny_owner = retain(&tiny).unwrap();
            let error = clamp::<$ty, 5>(&tiny_owner, &mut output).unwrap_err();
            let fault = error.recovered_range_fault().unwrap();
            assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
            assert_eq!(fault.invocation_id, 2);
            let mut recovered = expected;
            // Exact Neg preserves the representable subnormal payload; tightened underflow
            // policy reports recovery without inventing an extra rounding/minimum-normal lift.
            recovered[2] = <$ty>::from_bits($sign | 1);
            read(&output, &recovered, sentinel);
            let mut extended = retain(&[sentinel; 7]).unwrap();
            let mut extended_expected = [sentinel; 7];
            extended_expected[..5].copy_from_slice(&expected);
            neg::<$ty, 5>(&source, &mut extended).unwrap();
            fatal(&neg::<$ty, 5>(&invalid_owner, &mut extended));
            read_extended(&extended, &extended_expected, sentinel);
            let error = clamp::<$ty, 5>(&tiny_owner, &mut extended).unwrap_err();
            assert_eq!(error.recovered_range_fault().unwrap(), fault);
            extended_expected[..5].copy_from_slice(&recovered);
            read_extended(&extended, &extended_expected, sentinel);
            read(&sibling, &input, sentinel);
            neg::<$ty, 5>(&source, &mut output).unwrap();
            read(&output, &expected, sentinel);
            drop(source);
            drop(sibling);
            drop(invalid_owner);
            drop(tiny_owner);
            global::clear_thread_cache().unwrap();
            read(&output, &expected, sentinel);
        }
    }};
}
#[allow(clippy::cognitive_complexity)] // Four exact representations share the same borrow and publication matrix.
fn formats() {
    format!(PcuF16Bits, 0x8000, 0x3c00, 0x7e00);
    format!(PcuBf16Bits, 0x8000, 0x3f80, 0x7fc0);
    format!(PcuF8E4M3FnBits, 0x80, 0x38, 0x7f);
    format!(PcuF8E5M2Bits, 0x80, 0x3c, 0x7e);
    extended_output::<PcuF16Bits>(|value| PcuF16Bits::pcu_checked_from_f32(value).unwrap());
    extended_output::<PcuBf16Bits>(|value| PcuBf16Bits::pcu_checked_from_f32(value).unwrap());
    extended_output::<PcuF8E4M3FnBits>(|value| {
        PcuF8E4M3FnBits::pcu_checked_from_f32(value).unwrap()
    });
    extended_output::<PcuF8E5M2Bits>(|value| PcuF8E5M2Bits::pcu_checked_from_f32(value).unwrap());
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
                    formats();
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

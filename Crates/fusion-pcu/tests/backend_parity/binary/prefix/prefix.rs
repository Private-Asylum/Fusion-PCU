//! Retained binary inputs keep their full shapes; only logical prefixes participate.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuRangePolicy,
    PcuScalar,
    PcuTensor,
};

#[path = "source/source.rs"]
mod source;

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

trait PrefixSample: Sample {
    fn nonfinite() -> Self;
    fn tiny() -> Self;
    fn twice_tiny() -> Self;
}
macro_rules! sample {
    ($ty:ty, $nonfinite:expr) => {
        impl PrefixSample for $ty {
            fn nonfinite() -> Self {
                Self::from_bits($nonfinite)
            }
            fn tiny() -> Self {
                Self::from_bits(1)
            }
            fn twice_tiny() -> Self {
                Self::from_bits(2)
            }
        }
    };
}
sample!(PcuF16Bits, 0x7c00);
sample!(PcuBf16Bits, 0x7f80);
sample!(PcuF8E4M3FnBits, 0x7f);
sample!(PcuF8E5M2Bits, 0x7c);
sample!(f32, 0x7f80_0000);
sample!(f64, 0x7ff0_0000_0000_0000);

// These test-only matches call real generated functions with ordinary borrows.
macro_rules! call {
    ($normal:ident, $clamp:ident, $range:expr, $left:expr, $right:expr, $output:expr) => {
        match $range {
            PcuRangePolicy::Clamp => source::$clamp($left, $right, $output),
            _ => source::$normal($left, $right, $output),
        }
    };
}

fn check<T: Sample>(actual: &[T], expected: &[T; 7]) {
    bits(&actual[..7], expected);
    bits(&actual[7..], &vec![T::finite(16.0); actual.len() - 7]);
}

fn operations<T: PrefixSample>(
    left: &PcuTensor<T>,
    right: &PcuTensor<T>,
    raw: &[f32; 7],
    range: PcuRangePolicy,
) {
    let mut stack = [T::finite(16.0); 9];
    macro_rules! test {
        ($name:ident, $clamp:ident, $expected:expr) => {
            stack.fill(T::finite(16.0));
            call!($name, $clamp, range, left, right, &mut stack).unwrap();
            check(&stack, &$expected.map(T::finite));
        };
    }
    test!(add, clamp_add, raw.map(|x| x + 2.0));
    test!(sub, clamp_sub, raw.map(|x| x - 2.0));
    test!(mul, clamp_mul, raw.map(|x| x * 2.0));
    test!(div, clamp_div, raw.map(|x| x / 2.0));
    test!(grid, clamp_grid, raw.map(|x| x + 2.0));
    test!(left_broadcast, clamp_left_broadcast, [raw[0] + 2.0; 7]);
    test!(right_broadcast, clamp_right_broadcast, raw.map(|x| x + 2.0));
    // The second declaration is truly unused: zero capacity creates no input.
    call!(
        repeated,
        clamp_repeated,
        range,
        left,
        &[] as &[T],
        &mut stack
    )
    .unwrap();
    check(&stack, &raw.map(|x| T::finite(x * 2.0)));
}

fn mixed_and_owned<T: PrefixSample>(
    left: &PcuTensor<T>,
    right: &PcuTensor<T>,
    raw: &[f32; 7],
    range: PcuRangePolicy,
) -> PcuTensor<T> {
    let sentinel = T::finite(16.0);
    let host_left = raw.map(T::finite);
    let host_right = [T::finite(2.0); 7];
    let expected = raw.map(|x| T::finite(x + 2.0));
    let mut output = retain(&[sentinel; 13]).unwrap();
    let mut host = [sentinel; 15];
    // Each host-minimum/resident-full combination has its own cold input layout.
    call!(add, clamp_add, range, left, right, &mut output).unwrap();
    output.read_into(&mut host).unwrap();
    check(&host, &expected);
    call!(add, clamp_add, range, &host_left, right, &mut output).unwrap();
    output.read_into(&mut host).unwrap();
    check(&host, &expected);
    call!(add, clamp_add, range, left, &host_right, &mut output).unwrap();
    output.read_into(&mut host).unwrap();
    check(&host, &expected);
    call!(add, clamp_add, range, &host_left, &host_right, &mut output).unwrap();
    output.read_into(&mut host).unwrap();
    check(&host, &expected);
    let mut short = [sentinel; 6];
    assert!(call!(add, clamp_add, range, left, right, &mut short).is_err());
    bits(&short, &[sentinel; 6]);
    assert!(call!(add, clamp_add, range, &[sentinel; 6], right, &mut output).is_err());
    assert!(call!(add, clamp_add, range, left, &[sentinel; 6], &mut output).is_err());
    output.read_into(&mut host).unwrap();
    check(&host, &expected);
    assert_eq!(output.shape(), &[13]);
    output
}

fn failures<T: PrefixSample>(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    let sentinel = T::finite(16.0);
    let mut full = [T::finite(2.0); 11];
    full[0] = T::tiny();
    full[7..].fill(T::nonfinite());
    let left = retain(&full).unwrap();
    let right = retain(&full).unwrap();
    let mut output = [sentinel; 9];
    let result = call!(add, clamp_add, range, &left, &right, &mut output);
    if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
        assert!(
            matches!(result.unwrap_err(), PcuExecutionError::ArithmeticFault(fault)
            if fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow
                && fault.invocation_id == 0 && fault.recovered == (range == PcuRangePolicy::Clamp))
        );
    } else {
        result.unwrap();
    }
    if underflow != PcuFloatUnderflowPolicy::RejectSubnormalResult || range == PcuRangePolicy::Clamp
    {
        let mut expected = [T::finite(4.0); 7];
        expected[0] = T::twice_tiny();
        check(&output, &expected);
    } else {
        bits(&output, &[sentinel; 9]);
    }
    let before = output;
    full[2] = T::nonfinite();
    let bad = retain(&full).unwrap();
    let error = call!(add, clamp_add, range, &bad, &right, &mut output).unwrap_err();
    let (kind, lane) = if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult
        && range == PcuRangePolicy::Reject
    {
        (PcuExecutionFaultKind::ArithmeticUnderflow, 0)
    } else {
        (PcuExecutionFaultKind::InvalidFloatingOperand, 2)
    };
    assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == kind && fault.invocation_id == lane && !fault.recovered));
    bits(&output, &before);
    // A fresh division-by-zero is always fatal, even for explicitly clamped range.
    full[0] = T::finite(0.0);
    full[2] = T::finite(2.0);
    let zero = retain(&full).unwrap();
    let error = call!(div, clamp_div, range, &left, &zero, &mut output).unwrap_err();
    assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::DivideByZero && fault.invocation_id == 0 && !fault.recovered));
    bits(&output, &before);
    resident_failures([&left, &right], [&bad, &zero], &before, (kind, lane), range);
}

fn resident_failures<T: PrefixSample>(
    [left, right]: [&PcuTensor<T>; 2],
    [bad, zero]: [&PcuTensor<T>; 2],
    previous: &[T; 9],
    (kind, lane): (PcuExecutionFaultKind, u64),
    range: PcuRangePolicy,
) {
    let mut owner = retain(previous).unwrap();
    let error = call!(add, clamp_add, range, bad, right, &mut owner).unwrap_err();
    assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == kind && fault.invocation_id == lane && !fault.recovered));
    let mut host = [T::finite(16.0); 11];
    if super::super::publication::discarded_after_terminal_fault(&owner, &mut host, bits::<T>) {
        owner = retain(previous).unwrap();
    } else {
        bits(&host[..9], previous);
    }
    let error = call!(div, clamp_div, range, left, zero, &mut owner).unwrap_err();
    assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::DivideByZero && fault.invocation_id == 0 && !fault.recovered));
    host.fill(T::finite(16.0));
    if !super::super::publication::discarded_after_terminal_fault(&owner, &mut host, bits::<T>) {
        bits(&host[..9], previous);
        bits(&host[9..], &[T::finite(16.0); 2]);
    }
}

fn format<T: PrefixSample>(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    for phase in 1..=3_u8 {
        let raw = [0.5, 1.0, 2.0, 4.0, 1.0, 0.5, 2.0].map(|x| x * f32::from(phase));
        let mut full_left = [T::nonfinite(); 11];
        full_left[..7].copy_from_slice(&raw.map(T::finite));
        let mut full_right = [T::nonfinite(); 13];
        full_right[..7].fill(T::finite(2.0));
        let left = retain(&full_left).unwrap();
        let right = retain(&full_right).unwrap();
        operations(&left, &right, &raw, range);
        let survivor = mixed_and_owned(&left, &right, &raw, range);
        let mut host_left = [T::finite(16.0); 13];
        let mut host_right = [T::finite(16.0); 15];
        left.read_into(&mut host_left).unwrap();
        right.read_into(&mut host_right).unwrap();
        bits(&host_left[..11], &full_left);
        bits(&host_right[..13], &full_right);
        drop(left);
        drop(right);
        global::clear_thread_cache().unwrap();
        let mut host = [T::finite(16.0); 15];
        survivor.read_into(&mut host).unwrap();
        check(&host, &raw.map(|x| T::finite(x + 2.0)));
    }
    failures::<T>(underflow, range);
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for float_underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                global::configure(global::PcuExecutionPolicy {
                    backend,
                    numerical_mode,
                    float_underflow,
                    ..Default::default()
                })
                .unwrap();
                format::<PcuF16Bits>(float_underflow, range);
                format::<PcuBf16Bits>(float_underflow, range);
                format::<PcuF8E4M3FnBits>(float_underflow, range);
                format::<PcuF8E5M2Bits>(float_underflow, range);
                format::<f32>(float_underflow, range);
                format::<f64>(float_underflow, range);
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

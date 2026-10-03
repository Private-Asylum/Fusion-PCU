//! Full integer owners, minimum logical reads, and transactional range notices.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuNumericalMode,
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
    bits(&actual[7..], &vec![T::small(17); actual.len() - 7]);
}

fn operations<T: Sample>(
    left: &PcuTensor<T>,
    right: &PcuTensor<T>,
    raw: [u8; 7],
    range: PcuRangePolicy,
) {
    let mut host = [T::small(17); 9];
    macro_rules! test {
        ($name:ident, $clamp:ident, $expected:expr) => {
            call!($name, $clamp, range, left, right, &mut host).unwrap();
            check(&host, &$expected.map(T::small));
        };
    }
    test!(add, clamp_add, raw.map(|x| x + 2));
    test!(sub, clamp_sub, raw.map(|x| x - 2));
    test!(mul, clamp_mul, raw.map(|x| x * 2));
    test!(grid, clamp_grid, raw.map(|x| x + 2));
    test!(left_broadcast, clamp_left_broadcast, [raw[0] + 2; 7]);
    test!(right_broadcast, clamp_right_broadcast, raw.map(|x| x + 2));
    call!(
        repeated,
        clamp_repeated,
        range,
        left,
        &[] as &[T],
        &mut host
    )
    .unwrap();
    check(&host, &raw.map(|x| T::small(x * 2)));
}

fn mixed_and_owned<T: Sample>(
    left: &PcuTensor<T>,
    right: &PcuTensor<T>,
    raw: [u8; 7],
    range: PcuRangePolicy,
) -> PcuTensor<T> {
    let sentinel = T::small(17);
    let host_left = raw.map(T::small);
    let host_right = [T::small(2); 7];
    let expected = raw.map(|x| T::small(x + 2));
    let mut output = retain(&[sentinel; 13]).unwrap();
    let mut host = [sentinel; 15];
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
    assert!(call!(add, clamp_add, range, &[sentinel; 6], right, &mut output).is_err());
    assert!(call!(add, clamp_add, range, left, &[sentinel; 6], &mut output).is_err());
    output.read_into(&mut host).unwrap();
    check(&host, &expected);
    let mut short = [sentinel; 6];
    assert!(call!(add, clamp_add, range, left, right, &mut short).is_err());
    bits(&short, &[sentinel; 6]);
    assert_eq!(output.shape(), &[13]);
    output
}

fn range_fault<T: Sample>(kind: PcuExecutionFaultKind, range: PcuRangePolicy) {
    let extreme = if kind == PcuExecutionFaultKind::ArithmeticUnderflow {
        T::MIN
    } else {
        T::MAX
    };
    let mut full = [T::small(2); 11];
    full[2] = extreme;
    full[6] = extreme;
    // Endpoint suffixes exercise every high limb/sign bit without being read.
    full[7..].fill(T::MIN);
    let left = retain(&full).unwrap();
    let right = retain(&[T::small(1); 13]).unwrap();
    let mut output = retain(&[T::small(17); 13]).unwrap();
    let error = if kind == PcuExecutionFaultKind::ArithmeticUnderflow {
        call!(sub, clamp_sub, range, &left, &right, &mut output)
    } else {
        call!(add, clamp_add, range, &left, &right, &mut output)
    }
    .unwrap_err();
    assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == kind && fault.invocation_id == 2 && fault.recovered == (range == PcuRangePolicy::Clamp)));
    let mut host = [T::small(17); 15];
    let mut expected = [T::small(17); 7];
    if range == PcuRangePolicy::Clamp {
        expected.fill(T::small(
            if kind == PcuExecutionFaultKind::ArithmeticUnderflow {
                1
            } else {
                3
            },
        ));
        expected[2] = extreme;
        expected[6] = extreme;
    }
    if range == PcuRangePolicy::Clamp {
        output.read_into(&mut host).unwrap();
        check(&host, &expected);
    } else if super::super::publication::discarded_after_terminal_fault(
        &output, &mut host, bits::<T>,
    ) {
        output = retain(&[T::small(17); 13]).unwrap();
    } else {
        check(&host, &expected);
    }
    // Retry after a range notice must overwrite the prefix but preserve its tails.
    call!(
        add,
        clamp_add,
        range,
        &[T::small(2); 7],
        &right,
        &mut output
    )
    .unwrap();
    output.read_into(&mut host).unwrap();
    check(&host, &[T::small(3); 7]);
}

fn format<T: Sample>(range: PcuRangePolicy) {
    for phase in 1..=3_u8 {
        let raw = [2_u8, 3, 4, 5, 6, 7, 8].map(|x| x + phase);
        let mut full_left = [T::MAX; 11];
        full_left[..7].copy_from_slice(&raw.map(T::small));
        let mut full_right = [T::MIN; 13];
        full_right[..7].fill(T::small(2));
        let left = retain(&full_left).unwrap();
        let right = retain(&full_right).unwrap();
        operations(&left, &right, raw, range);
        let survivor = mixed_and_owned(&left, &right, raw, range);
        let mut host_left = [T::small(17); 13];
        let mut host_right = [T::small(17); 15];
        left.read_into(&mut host_left).unwrap();
        right.read_into(&mut host_right).unwrap();
        bits(&host_left[..11], &full_left);
        bits(&host_right[..13], &full_right);
        drop(left);
        drop(right);
        global::clear_thread_cache().unwrap();
        let mut host = [T::small(17); 15];
        survivor.read_into(&mut host).unwrap();
        check(&host, &raw.map(|x| T::small(x + 2)));
    }
    range_fault::<T>(PcuExecutionFaultKind::ArithmeticOverflow, range);
    range_fault::<T>(PcuExecutionFaultKind::ArithmeticUnderflow, range);
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            global::configure(global::PcuExecutionPolicy {
                backend,
                numerical_mode,
                ..Default::default()
            })
            .unwrap();
            format::<u8>(range);
            format::<i8>(range);
            format::<u16>(range);
            format::<i16>(range);
            format::<u32>(range);
            format::<i32>(range);
            format::<u64>(range);
            format::<i64>(range);
            format::<u128>(range);
            format::<i128>(range);
            format::<PcuU256>(range);
            format::<PcuI256>(range);
            format::<PcuU512>(range);
            format::<PcuI512>(range);
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

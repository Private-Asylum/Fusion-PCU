#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedIntegerDivision,
    PcuCompoundArithmeticPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
};
#[rustfmt::skip]
use super::{oracle,source};
use oracle::Wide;

fn invoke<T: PcuCheckedIntegerDivision>(
    kind: usize,
    left: &[T],
    right: &[T],
    q: &mut [T],
    r: &mut [T],
) -> Result<(), global::PcuExecutionError> {
    match kind {
        0 => source::repeated::<T, 5>(q, r, left),
        1 => source::unused::<T, 5>(&[], left, r, q),
        2 => source::reordered::<T, 5>(right, r, left, q),
        3 => source::mixed::<T, 5>(q, left, r),
        4 => source::grid::<T, 5>(r, &[], q, left),
        _ => source::scalar::<T, 5>(&left[0], q, &right[0], r),
    }
}
const fn operands<T: Copy>(kind: usize, left: &[T], right: &[T], lane: usize) -> (T, T) {
    match kind {
        0 | 1 => (left[lane], left[lane]),
        2 => (left[lane], right[lane]),
        3 => (left[lane], left[0]),
        4 => (left[0], left[lane]),
        _ => (left[0], right[0]),
    }
}
fn verify_call<T: Wide + PcuCheckedIntegerDivision>(kind: usize, left: &[T], right: &[T]) {
    let sentinel = oracle::small::<T>(99);
    let mut q = [sentinel; 8];
    let mut r = q;
    let mut expected_q = q;
    let mut expected_r = r;
    let mut fault = None;
    for lane in 0..5 {
        let (a, b) = operands(kind, left, right, lane);
        match oracle::evaluate(a, b) {
            Ok((quotient, remainder)) => {
                expected_q[lane] = quotient;
                expected_r[lane] = remainder;
            }
            Err(error) => {
                fault = Some((lane as u64, error));
                break;
            }
        }
    }
    let result = invoke(kind, left, right, &mut q, &mut r);
    if let Some((lane, kind)) = fault {
        let actual = result.unwrap_err().arithmetic_fault().unwrap();
        assert_eq!(
            (actual.invocation_id, actual.kind, actual.recovered),
            (lane, kind, false)
        );
        assert_eq!((q, r), ([sentinel; 8], [sentinel; 8]));
    } else {
        result.unwrap();
        assert_eq!((q, r), (expected_q, expected_r));
    }
}
fn verify<T: Wide + PcuCheckedIntegerDivision>() {
    let three = oracle::small(3);
    let seven = oracle::small(7);
    let zero = oracle::small(0);
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for underflow in [
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                ] {
                    global::configure(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Cpu,
                        numerical_mode: mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            ..Default::default()
                        },
                        float_underflow: underflow,
                        ..Default::default()
                    })
                    .unwrap();
                    for kind in 0..6 {
                        let mut left = [seven; 5];
                        left[0] = three;
                        left[2] = oracle::maximum::<T>();
                        let mut right = [three; 5];
                        for _ in 0..3 {
                            left[3] = if left[3] == seven {
                                oracle::maximum::<T>()
                            } else {
                                seven
                            };
                            verify_call(kind, &left, &right);
                        }
                        if kind == 2 || kind == 5 {
                            right[if kind == 5 { 0 } else { 2 }] = zero;
                        } else {
                            left[if kind == 3 { 0 } else { 2 }] = zero;
                        }
                        verify_call(kind, &left, &right);
                        right.fill(three);
                        left.fill(seven);
                        verify_call(kind, &left, &right);
                        let sentinel = oracle::small::<T>(99);
                        let mut q = [sentinel; 8];
                        let mut r = q;
                        assert!(invoke(kind, &left, &right, &mut q, &mut r[..4]).is_err());
                        assert_eq!((q, r), ([sentinel; 8], [sentinel; 8]));
                        invoke(kind, &left, &right, &mut q, &mut r).unwrap();
                        if T::SIGNED && (2..=5).contains(&kind) {
                            let minus_one = T::from_bytes([255; 64]);
                            if kind == 3 {
                                left[0] = minus_one;
                                left[2] = oracle::minimum();
                            } else if kind == 4 {
                                left[0] = oracle::minimum();
                                left[2] = minus_one;
                            } else {
                                left[if kind == 5 { 0 } else { 2 }] = oracle::minimum();
                                right[if kind == 5 { 0 } else { 2 }] = minus_one;
                            }
                            verify_call(kind, &left, &right);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fourteen_formats_local_portable_source_faults_and_retry() {
    macro_rules! formats {($($ty:ty),+)=>{$(verify::<$ty>();)+};}
    formats!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        pcu_facade::PcuI256,
        pcu_facade::PcuU256,
        pcu_facade::PcuI512,
        pcu_facade::PcuU512
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

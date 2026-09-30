//! Checked integer pointwise execution through the owned tensor source API.
#![cfg(all(feature = "rocm", feature = "tensor"))]

use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCheckedInteger,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuScalar,
    PcuTensor,
};

static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[pcu]
fn add_u8(lhs: &[u8], rhs: &[u8]) -> Result<PcuTensor<u8>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn discarded_add_u8(lhs: &[u8], rhs: &[u8]) -> Result<PcuTensor<u8>, PcuExecutionError> {
    let _unused = lhs + rhs;
    Ok(pcu::identity(lhs)?)
}

#[pcu]
fn sub_u8(lhs: &[u8], rhs: &[u8]) -> Result<PcuTensor<u8>, PcuExecutionError> {
    Ok(lhs - rhs)
}

#[pcu]
fn mul_u8(lhs: &[u8], rhs: &[u8]) -> Result<PcuTensor<u8>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu]
fn masked_u8(lhs: &[u8], rhs: &[u8], zero: &[u8]) -> Result<PcuTensor<u8>, PcuExecutionError> {
    Ok(pcu::mul(&(lhs + rhs), zero)?)
}

#[pcu]
fn add_u16(lhs: &[u16], rhs: &[u16]) -> Result<PcuTensor<u16>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn sub_u16(lhs: &[u16], rhs: &[u16]) -> Result<PcuTensor<u16>, PcuExecutionError> {
    Ok(lhs - rhs)
}

#[pcu]
fn mul_u16(lhs: &[u16], rhs: &[u16]) -> Result<PcuTensor<u16>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu]
fn masked_u16(lhs: &[u16], rhs: &[u16], zero: &[u16]) -> Result<PcuTensor<u16>, PcuExecutionError> {
    Ok(pcu::mul(&(lhs + rhs), zero)?)
}

#[pcu]
fn add_u32(lhs: &[u32], rhs: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn sub_u32(lhs: &[u32], rhs: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    Ok(lhs - rhs)
}

#[pcu]
fn mul_u32(lhs: &[u32], rhs: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu]
fn masked_u32(lhs: &[u32], rhs: &[u32], zero: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    Ok(pcu::mul(&(lhs + rhs), zero)?)
}

#[pcu]
fn add_u64(lhs: &[u64], rhs: &[u64]) -> Result<PcuTensor<u64>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn sub_u64(lhs: &[u64], rhs: &[u64]) -> Result<PcuTensor<u64>, PcuExecutionError> {
    Ok(lhs - rhs)
}

#[pcu]
fn mul_u64(lhs: &[u64], rhs: &[u64]) -> Result<PcuTensor<u64>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu]
fn masked_u64(lhs: &[u64], rhs: &[u64], zero: &[u64]) -> Result<PcuTensor<u64>, PcuExecutionError> {
    Ok(pcu::mul(&(lhs + rhs), zero)?)
}

#[pcu]
fn add_i8(lhs: &[i8], rhs: &[i8]) -> Result<PcuTensor<i8>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn sub_i8(lhs: &[i8], rhs: &[i8]) -> Result<PcuTensor<i8>, PcuExecutionError> {
    Ok(lhs - rhs)
}

#[pcu]
fn mul_i8(lhs: &[i8], rhs: &[i8]) -> Result<PcuTensor<i8>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu]
fn masked_i8(lhs: &[i8], rhs: &[i8], zero: &[i8]) -> Result<PcuTensor<i8>, PcuExecutionError> {
    Ok(pcu::mul(&(lhs + rhs), zero)?)
}

#[pcu]
fn add_i16(lhs: &[i16], rhs: &[i16]) -> Result<PcuTensor<i16>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn sub_i16(lhs: &[i16], rhs: &[i16]) -> Result<PcuTensor<i16>, PcuExecutionError> {
    Ok(lhs - rhs)
}

#[pcu]
fn mul_i16(lhs: &[i16], rhs: &[i16]) -> Result<PcuTensor<i16>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu]
fn masked_i16(lhs: &[i16], rhs: &[i16], zero: &[i16]) -> Result<PcuTensor<i16>, PcuExecutionError> {
    Ok(pcu::mul(&(lhs + rhs), zero)?)
}

#[pcu]
fn add_i32(lhs: &[i32], rhs: &[i32]) -> Result<PcuTensor<i32>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn sub_i32(lhs: &[i32], rhs: &[i32]) -> Result<PcuTensor<i32>, PcuExecutionError> {
    Ok(lhs - rhs)
}

#[pcu]
fn mul_i32(lhs: &[i32], rhs: &[i32]) -> Result<PcuTensor<i32>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu]
fn masked_i32(lhs: &[i32], rhs: &[i32], zero: &[i32]) -> Result<PcuTensor<i32>, PcuExecutionError> {
    Ok(pcu::mul(&(lhs + rhs), zero)?)
}

#[pcu]
fn add_i64(lhs: &[i64], rhs: &[i64]) -> Result<PcuTensor<i64>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn sub_i64(lhs: &[i64], rhs: &[i64]) -> Result<PcuTensor<i64>, PcuExecutionError> {
    Ok(lhs - rhs)
}

#[pcu]
fn mul_i64(lhs: &[i64], rhs: &[i64]) -> Result<PcuTensor<i64>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu]
fn masked_i64(lhs: &[i64], rhs: &[i64], zero: &[i64]) -> Result<PcuTensor<i64>, PcuExecutionError> {
    Ok(pcu::mul(&(lhs + rhs), zero)?)
}

trait IntegerCase: PcuCheckedInteger + Copy + core::fmt::Debug + PartialEq {
    const MIN: Self;
    const MAX: Self;
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEGATIVE_ONE: Option<Self>;
}

macro_rules! impl_integer_case {
    ($ty:ty, $negative_one:expr) => {
        impl IntegerCase for $ty {
            const MIN: Self = <$ty>::MIN;
            const MAX: Self = <$ty>::MAX;
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const TWO: Self = 2;
            const NEGATIVE_ONE: Option<Self> = $negative_one;
        }
    };
}

impl_integer_case!(u8, None);
impl_integer_case!(u16, None);
impl_integer_case!(u32, None);
impl_integer_case!(u64, None);
impl_integer_case!(i8, Some(-1));
impl_integer_case!(i16, Some(-1));
impl_integer_case!(i32, Some(-1));
impl_integer_case!(i64, Some(-1));

type BinarySource<T> = fn(&[T], &[T]) -> Result<PcuTensor<T>, PcuExecutionError>;
type MaskedSource<T> = fn(&[T], &[T], &[T]) -> Result<PcuTensor<T>, PcuExecutionError>;

fn assert_fault<T: PcuScalar + core::fmt::Debug>(
    result: Result<PcuTensor<T>, PcuExecutionError>,
    kind: PcuExecutionFaultKind,
    element: u64,
) {
    match result {
        Err(PcuExecutionError::ArithmeticFault(fault))
            if fault.kind == kind && fault.invocation_id == element => {}
        other => panic!("expected {kind:?} at tensor element {element}, got {other:?}"),
    }
}

fn read_output<T: IntegerCase>(owner: &PcuTensor<T>, expected: &[T]) {
    let mut actual = expected.to_vec();
    owner.read_into(&mut actual).unwrap();
    assert_eq!(actual, expected);
}

fn exercise_type<T: IntegerCase>(
    add: BinarySource<T>,
    sub: BinarySource<T>,
    mul: BinarySource<T>,
    masked: MaskedSource<T>,
) {
    let zeros = [T::ZERO, T::ZERO];
    let ones = [T::ONE, T::ONE];
    let add_lhs = [T::ZERO, T::MAX];
    assert_fault(
        add(&add_lhs, &ones),
        PcuExecutionFaultKind::ArithmeticOverflow,
        1,
    );

    let sub_lhs = [T::ONE, T::MIN];
    assert_fault(
        sub(&sub_lhs, &ones),
        PcuExecutionFaultKind::ArithmeticUnderflow,
        1,
    );

    let mul_lhs = [T::ONE, T::MAX];
    let mul_rhs = [T::ONE, T::TWO];
    assert_fault(
        mul(&mul_lhs, &mul_rhs),
        PcuExecutionFaultKind::ArithmeticOverflow,
        1,
    );

    assert_fault(
        masked(&add_lhs, &ones, &zeros),
        PcuExecutionFaultKind::ArithmeticOverflow,
        1,
    );

    if let Some(negative_one) = T::NEGATIVE_ONE {
        let underflow_lhs = [T::ZERO, T::MIN];
        let underflow_rhs = [T::ZERO, negative_one];
        assert_fault(
            add(&underflow_lhs, &underflow_rhs),
            PcuExecutionFaultKind::ArithmeticUnderflow,
            1,
        );
        let overflow_lhs = [T::ZERO, T::MAX];
        let overflow_rhs = [T::ZERO, negative_one];
        assert_fault(
            sub(&overflow_lhs, &overflow_rhs),
            PcuExecutionFaultKind::ArithmeticOverflow,
            1,
        );
        let underflow_lhs = [T::ZERO, T::MIN];
        assert_fault(
            mul(&underflow_lhs, &mul_rhs),
            PcuExecutionFaultKind::ArithmeticUnderflow,
            1,
        );
    }

    // A fault does not poison the captured source function; changed warm inputs run successfully.
    let left = [T::ONE, T::TWO];
    let right = [T::TWO, T::ONE];
    let warm = add(&left, &right).unwrap();
    let expected = [
        T::ONE.pcu_checked_add(T::TWO).unwrap(),
        T::TWO.pcu_checked_add(T::ONE).unwrap(),
    ];
    read_output(&warm, &expected);

    // The API borrows input slices, so arithmetic faults cannot mutate caller storage.
    assert_eq!(add_lhs, [T::ZERO, T::MAX]);
    assert_eq!(ones, [T::ONE, T::ONE]);
    global::clear_thread_cache().unwrap();
    let after_reset = add(&left, &right).unwrap();
    read_output(&after_reset, &expected);
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn owned_checked_integer_add_sub_mul_faults_are_typed_terminal_and_retryable() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    // The selected program must retain a checked operation even when its value is discarded.
    assert_fault(
        discarded_add_u8(&[1, u8::MAX], &[1, 1]),
        PcuExecutionFaultKind::ArithmeticOverflow,
        1,
    );

    exercise_type::<u8>(add_u8, sub_u8, mul_u8, masked_u8);
    exercise_type::<u16>(add_u16, sub_u16, mul_u16, masked_u16);
    exercise_type::<u32>(add_u32, sub_u32, mul_u32, masked_u32);
    exercise_type::<u64>(add_u64, sub_u64, mul_u64, masked_u64);
    exercise_type::<i8>(add_i8, sub_i8, mul_i8, masked_i8);
    exercise_type::<i16>(add_i16, sub_i16, mul_i16, masked_i16);
    exercise_type::<i32>(add_i32, sub_i32, mul_i32, masked_i32);
    exercise_type::<i64>(add_i64, sub_i64, mul_i64, masked_i64);

    global::clear_thread_cache().unwrap();
}

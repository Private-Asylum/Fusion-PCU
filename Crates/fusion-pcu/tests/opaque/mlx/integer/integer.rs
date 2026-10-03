//! Genuine fourteen-width owned calls, exact checked effects and escaped-owner lifetimes.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedInteger,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu]
fn add<T: PcuCheckedInteger>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}
#[pcu]
fn sub<T: PcuCheckedInteger>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sub(right, left)
}
#[pcu]
fn mul<T: PcuCheckedInteger>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(left, right)
}
#[pcu]
fn repeated<T: PcuCheckedInteger>(
    unused: &[T],
    input: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(input, input)
}
#[pcu]
fn checked_unused<T: PcuCheckedInteger>(
    left: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _checked = pcu::add(left, right);
    pcu::identity(left)
}
#[pcu(flag(strict))]
fn local_strict<T: PcuCheckedInteger>(
    left: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}
#[pcu]
fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
trait Sample: PcuCheckedInteger {
    fn small(value: u8) -> Self;
    fn maximum() -> Self;
}
macro_rules! sample {
    ($($ty:ty=>$width:literal;)+) => {$(impl Sample for $ty {
        fn small(value:u8)->Self {let mut bytes=[0;$width];bytes[0]=value;Self::decode_le(bytes)}
        fn maximum()->Self {Self::MAX}
    })+};
}
sample! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
fn read<T: Sample>(owner: &PcuTensor<T>, expected: &[T; 5]) {
    let sentinel = T::small(19);
    let mut stack = [sentinel; 7];
    assert_eq!(owner.shape(), [5]);
    owner.read_into(&mut stack).unwrap();
    for (actual, expected) in stack
        .iter()
        .zip(expected.iter().chain([sentinel; 2].iter()))
    {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    let mut short = [sentinel; 4];
    assert!(owner.read_into(&mut short).is_err());
    for value in short {
        assert_eq!(value.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}
fn fault<T: PcuScalar>(result: Result<PcuTensor<T>, PcuExecutionError>) {
    let error = result.unwrap_err();
    let observed = error.arithmetic_fault().unwrap();
    assert_eq!(observed.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(observed.invocation_id, 1);
    assert!(!observed.recovered);
}
fn roles<T: Sample>() {
    for phase in 1..=3 {
        let left = [T::small(phase); 5];
        let right = [T::small(phase * 2); 5];
        let sum = left[0].pcu_checked_add(right[0]).unwrap();
        let difference = right[0].pcu_checked_sub(left[0]).unwrap();
        let product = left[0].pcu_checked_mul(right[0]).unwrap();
        let square = right[0].pcu_checked_mul(right[0]).unwrap();
        let a = retain(&left).unwrap();
        let b = retain(&right).unwrap();
        read(&add(&left, &right).unwrap(), &[sum; 5]);
        read(&add::<T>(&a, &right).unwrap(), &[sum; 5]);
        read(&add::<T>(&left, &b).unwrap(), &[sum; 5]);
        let escaped = add::<T>(&a, &b).unwrap();
        read(&escaped, &[sum; 5]);
        read(&sub::<T>(&a, &b).unwrap(), &[difference; 5]);
        read(&mul::<T>(&a, &b).unwrap(), &[product; 5]);
        // The detached-capture helper has a separate zero-shape gate; this
        // ordinary path must prune unused host declarations itself.
        read(&repeated::<T>(&[T::small(0)], &b).unwrap(), &[square; 5]);
        read(&repeated::<T>(&[], &b).unwrap(), &[square; 5]);
        read(&checked_unused::<T>(&a, &b).unwrap(), &left);
        read(&local_strict::<T>(&a, &b).unwrap(), &[sum; 5]);
        let mut bad = [T::small(0); 5];
        bad[1] = T::maximum();
        bad[3] = T::maximum();
        let bad_owner = retain(&bad).unwrap();
        fault(add::<T>(&a, &bad_owner));
        fault(checked_unused::<T>(&a, &bad_owner));
        read(&a, &left);
        read(&bad_owner, &bad);
        read(&escaped, &[sum; 5]);
        let retry = checked_unused::<T>(&a, &b).unwrap();
        let sibling = consume(escaped).unwrap();
        global::clear_thread_cache().unwrap();
        drop(a);
        drop(b);
        drop(bad_owner);
        read(&retry, &left);
        read(&sibling, &[sum; 5]);
        drop(retry);
        read(&sibling, &[sum; 5]);
    }
}
fn all_types() {
    roles::<u8>();
    roles::<i8>();
    roles::<u16>();
    roles::<i16>();
    roles::<u32>();
    roles::<i32>();
    roles::<u64>();
    roles::<i64>();
    roles::<u128>();
    roles::<i128>();
    roles::<PcuU256>();
    roles::<PcuI256>();
    roles::<PcuU512>();
    roles::<PcuI512>();
}
#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires actual MLX fourteen-width owned source and retained native sessions"
)]
fn ordinary_mlx_fourteen_width_owned_integer_roles_effects_and_lifetimes() {
    let _guard = super::POLICY_LOCK.lock().unwrap();
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
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    let mut options = global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Mlx,
                        numerical_mode: mode,
                        float_underflow: underflow,
                        ..Default::default()
                    };
                    options.numerical_options.compound_arithmetic = compound;
                    options.numerical_options.precision = precision;
                    global::configure(options).unwrap();
                    all_types();
                }
            }
        }
    }
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        range_policy: PcuRangePolicy::Clamp,
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        add(&[1_u8; 5], &[1_u8; 5]),
        Err(PcuExecutionError::UnsupportedRangePolicy)
    ));
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

//! Ordinary annotated calls retain exact MLX owners across host/resident roles.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuCompoundArithmeticPolicy,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuScalar,
    PcuTensor,
    global::{
        PcuBackendChoice,
        PcuExecutionPolicy,
    },
};

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu]
fn add<T: PcuCheckedFloat>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}
#[pcu]
fn sub<T: PcuCheckedFloat>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sub(right, left)
}
#[pcu]
fn mul<T: PcuCheckedFloat>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(left, right)
}
#[pcu]
fn div<T: PcuCheckedFloat>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::div(right, left)
}
#[pcu]
fn repeated<T: PcuCheckedFloat>(
    _unused: &[T],
    input: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(input, input)
}
#[pcu]
fn checked_unused<T: PcuCheckedFloat>(
    left: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _checked = pcu::div(right, left);
    pcu::identity(right)
}

trait Sample: PcuCheckedFloat {
    fn finite(input: f32) -> Self;
}

macro_rules! low_samples {
    ($($ty:ty),+) => {$(
        impl Sample for $ty {
            fn finite(input: f32) -> Self {
                Self::pcu_checked_from_f32(input).unwrap()
            }
        }
    )+};
}
low_samples!(
    fusion_pcu::PcuF16Bits,
    fusion_pcu::PcuBf16Bits,
    fusion_pcu::PcuF8E4M3FnBits,
    fusion_pcu::PcuF8E5M2Bits
);
impl Sample for f32 {
    fn finite(input: f32) -> Self {
        input
    }
}
impl Sample for f64 {
    fn finite(input: f32) -> Self {
        Self::from(input)
    }
}

fn value<T: Sample>(input: f32) -> T {
    T::finite(input)
}

fn read<T: Sample>(owner: &PcuTensor<T>, expected: &[T; 5]) {
    let sentinel = value::<T>(16.0);
    let mut stack = [sentinel; 7];
    owner.read_into(&mut stack).unwrap();
    assert_eq!(owner.shape(), [5]);
    for (actual, expected) in stack
        .iter()
        .zip(expected.iter().chain([sentinel; 2].iter()))
    {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    let mut short = [sentinel; 4];
    assert!(owner.read_into(&mut short).is_err());
    for actual in short {
        assert_eq!(actual.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}

fn fault<T: PcuScalar>(result: Result<PcuTensor<T>, PcuExecutionError>) {
    let error = result.unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::DivideByZero);
    assert_eq!(fault.invocation_id, 2);
    assert!(!fault.recovered);
}

fn roles<T: Sample>() {
    let left = [value::<T>(2.0); 5];
    let right = [value::<T>(4.0); 5];
    let a = retain(&left).unwrap();
    let b = retain(&right).unwrap();
    // The same source site changes physical roles. Only an explicit read_into
    // below crosses back to RAM; borrowed owners keep their original session.
    read(&add(&left, &right).unwrap(), &[value::<T>(6.0); 5]);
    read(&add::<T>(&a, &right).unwrap(), &[value::<T>(6.0); 5]);
    read(&add::<T>(&left, &b).unwrap(), &[value::<T>(6.0); 5]);
    let output = add::<T>(&a, &b).unwrap();
    read(&output, &[value::<T>(6.0); 5]);
    read(&sub::<T>(&a, &b).unwrap(), &left);
    read(&mul::<T>(&a, &b).unwrap(), &[value::<T>(8.0); 5]);
    read(&div::<T>(&a, &b).unwrap(), &left);
    // Parameter zero is physically unused, even though parameter one is an
    // escaped device owner; repeated operands produce just one actual binding.
    read(&repeated::<T>(&[], &b).unwrap(), &[value::<T>(16.0); 5]);
    read(&checked_unused::<T>(&a, &b).unwrap(), &right);
    let mut zeros = left;
    zeros[2] = value::<T>(0.0);
    zeros[4] = value::<T>(0.0);
    let bad = retain(&zeros).unwrap();
    fault(div::<T>(&bad, &b));
    fault(checked_unused::<T>(&bad, &b));
    read(&bad, &zeros);
    read(&output, &[value::<T>(6.0); 5]);
    let retry = div::<T>(&a, &b).unwrap();
    global::clear_thread_cache().unwrap();
    drop(a);
    drop(b);
    drop(bad);
    read(&retry, &left);
    read(&output, &[value::<T>(6.0); 5]);
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires actual Apple silicon six-format MLX owned binary source"
)]
fn ordinary_mlx_six_format_owned_binary_roles_and_faults() {
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
                    let mut options = PcuExecutionPolicy {
                        backend: PcuBackendChoice::Mlx,
                        numerical_mode: mode,
                        float_underflow: underflow,
                        ..Default::default()
                    };
                    options.numerical_options.compound_arithmetic = compound;
                    options.numerical_options.precision = precision;
                    global::configure(options).unwrap();
                    roles::<fusion_pcu::PcuF16Bits>();
                    roles::<fusion_pcu::PcuBf16Bits>();
                    roles::<fusion_pcu::PcuF8E4M3FnBits>();
                    roles::<fusion_pcu::PcuF8E5M2Bits>();
                    roles::<f32>();
                    roles::<f64>();
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

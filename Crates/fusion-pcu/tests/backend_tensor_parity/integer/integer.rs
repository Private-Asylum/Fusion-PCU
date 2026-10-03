//! Full-width tensor pointwise source, escaping owners, borrow stability and checked failure.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedInteger,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuI256,
    PcuI512,
    PcuNumericalMode,
    PcuScalar,
    PcuTensor,
    PcuU256,
    PcuU512,
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
fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

trait Sample: PcuCheckedInteger {
    const MAX: Self;
    const MIN: Self;
    fn small(value: u8) -> Self;
}

macro_rules! native {
    ($($ty:ty),+ $(,)?) => {$(
        impl Sample for $ty {
            const MAX: Self = Self::MAX;
            const MIN: Self = Self::MIN;
            fn small(value: u8) -> Self { Self::try_from(value).unwrap() }
        }
    )+};
}
native!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);

macro_rules! wide {
    ($ty:ty, $width:literal, $minimum:expr) => {
        impl Sample for $ty {
            const MAX: Self = Self::MAX;
            const MIN: Self = $minimum;
            fn small(value: u8) -> Self {
                let mut limbs = [0; $width];
                limbs[0] = u64::from(value);
                Self::from_limbs_le(limbs)
            }
        }
    };
}
wide!(PcuU256, 4, Self::ZERO);
wide!(PcuI256, 4, Self::MIN);
wide!(PcuU512, 8, Self::ZERO);
wide!(PcuI512, 8, Self::MIN);

fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (left, right) in actual.iter().zip(expected) {
        assert_eq!(left.encode_le().as_ref(), right.encode_le().as_ref());
    }
}

fn output<T: Sample>(owner: &PcuTensor<T>, expected: &[T; 7]) {
    assert_eq!(owner.shape(), &[7]);
    let mut stack = [T::small(97); 9];
    owner.read_into(&mut stack).unwrap();
    bits(&stack[..7], expected);
    bits(&stack[7..], &[T::small(97); 2]);
}

fn fault<T: PcuScalar>(
    result: Result<PcuTensor<T>, PcuExecutionError>,
    kind: PcuExecutionFaultKind,
) {
    let Err(error) = result else {
        panic!("checked range exception returned a usable owner")
    };
    match error {
        PcuExecutionError::ArithmeticFault(fault) => {
            assert_eq!(fault.kind, kind);
            assert_eq!(fault.invocation_id, 2);
            assert!(!fault.recovered);
        }
        // The tensor reference retains its graph node identifier in addition to the lane.
        PcuExecutionError::TensorBuild(
            fusion_pcu::dialect::tensor::TensorError::ArithmeticFault {
                element_index,
                kind: actual,
                ..
            },
        ) => {
            assert_eq!(actual, kind);
            assert_eq!(element_index, 2);
        }
        other => panic!("expected checked {kind:?} at lane2, received {other:?}"),
    }
}

fn format<T: Sample>() {
    global::clear_thread_cache().unwrap();
    for phase in [0_u8, 1, 2] {
        let raw = [3_u8, 4, 5, 6, 7, 8, 9].map(|value| value + phase);
        let lhs = raw.map(T::small);
        let rhs = [T::small(2); 7];
        let a = retain(&lhs).unwrap();
        let b = retain(&rhs).unwrap();
        let sum = add::<T>(&a, &b).unwrap();
        let restored = sub::<T>(&sum, &b).unwrap();
        let product = mul::<T>(&restored, &b).unwrap();
        output(&a, &lhs);
        output(&b, &rhs);
        output(&sum, &raw.map(|value| T::small(value + 2)));
        output(&restored, &lhs);
        drop(a);
        drop(sum);
        drop(restored);
        output(&product, &raw.map(|value| T::small(value * 2)));
        let moved = consume(product).unwrap();
        output(&moved, &raw.map(|value| T::small(value * 2)));

        let mut exceptional = lhs;
        exceptional[2] = T::MAX;
        exceptional[6] = T::MAX;
        let bad = retain(&exceptional).unwrap();
        fault(
            add::<T>(&bad, &[T::small(1); 7]),
            PcuExecutionFaultKind::ArithmeticOverflow,
        );
        fault(
            mul::<T>(&bad, &b),
            PcuExecutionFaultKind::ArithmeticOverflow,
        );
        output(&bad, &exceptional);
        exceptional[2] = T::MIN;
        exceptional[6] = T::MIN;
        fault(
            sub::<T>(&exceptional, &[T::small(1); 7]),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        // A failed private result cannot invalidate the borrowed input or an escaped sibling.
        output(&b, &rhs);
        output(&moved, &raw.map(|value| T::small(value * 2)));
        let retry = add::<T>(&lhs, &b).unwrap();
        output(&retry, &raw.map(|value| T::small(value + 2)));
    }
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        global::configure(global::PcuExecutionPolicy {
            backend,
            numerical_mode,
            ..Default::default()
        })
        .unwrap();
        format::<u8>();
        format::<i8>();
        format::<u16>();
        format::<i16>();
        format::<u32>();
        format::<i32>();
        format::<u64>();
        format::<i64>();
        format::<u128>();
        format::<i128>();
        format::<PcuU256>();
        format::<PcuI256>();
        format::<PcuU512>();
        format::<PcuI512>();
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[cfg(feature = "cpu")]
#[path = "invocation/invocation.rs"]
mod invocation;

#[cfg(feature = "cpu")]
#[test]
fn cpu_owned_integer_invocation_borrows_preserve_affinity_tails_and_results() {
    invocation::verify();
}

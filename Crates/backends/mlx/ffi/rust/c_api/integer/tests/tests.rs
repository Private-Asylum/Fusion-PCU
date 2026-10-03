//! Independent checked/clamped core integer oracle for the distinct MLX-owned primitive.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedInteger,
    PcuScalar,
    PcuBindingRef,
    PcuHostArgument,
    PcuExecutionFaultKind as Kind,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
trait Sample: PcuCheckedInteger {
    fn raw(bytes: &[u8]) -> Self;
    fn zero() -> Self;
}
macro_rules! sample {
    ($($ty:ty => $width:literal;)+) => {$(
        impl Sample for $ty {
            fn zero() -> Self { Self::decode_le([0; $width]) }
            fn raw(bytes: &[u8]) -> Self {
                Self::decode_le(<[u8; $width]>::try_from(bytes).unwrap())
            }
        }
    )+};
}
sample! {
    u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;
    u128=>16;i128=>16;PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;
}
fn expected<T: Sample>(left: T, right: T, operation: u32, range: Range) -> (T, u32) {
    let zero = T::zero();
    let result = match operation {
        0 => left.pcu_clamped_add(right),
        1 => left.pcu_clamped_sub(right),
        _ => left.pcu_clamped_mul(right),
    };
    match result {
        Ok(value) => (value, 0),
        Err(fault) => {
            let kind = match fault.kind() {
                Kind::ArithmeticOverflow => 1,
                Kind::ArithmeticUnderflow => 3,
                other => panic!("unexpected integer oracle fault {other:?}"),
            };
            if range == Range::Clamp {
                (fault.clamped_value(), kind | 0x100)
            } else {
                (zero, kind)
            }
        }
    }
}
fn qualify<T: Sample>(session: &Session, left: &[T], right: &[T], broadcast: [bool; 2]) {
    let count = left.len().max(right.len());
    let a = PcuHostArgument::read(PcuBindingRef::new(0, 0), left);
    let b = PcuHostArgument::read(PcuBindingRef::new(0, 1), right);
    let a = EncodedArray::upload(session, T::TYPE, left.len(), a.bytes()).unwrap();
    let b = EncodedArray::upload(session, T::TYPE, right.len(), b.bytes()).unwrap();
    for range in [Range::Reject, Range::Clamp] {
        for operation in 0..3 {
            let prepared = CheckedInteger::prepare(
                session,
                T::TYPE,
                operation,
                range,
                count,
                [left.len(), right.len()],
                broadcast,
            )
            .unwrap();
            let (payload, records) = prepared.execute_recorded([&a, &b]).unwrap();
            let statuses = prepared.status_records(&records).unwrap();
            let mut bytes = vec![0; count * T::ENCODED_SIZE];
            payload.read(&mut bytes).unwrap();
            for (index, (value, &status)) in bytes.chunks(T::ENCODED_SIZE).zip(statuses).enumerate()
            {
                let lhs = left[if broadcast[0] { 0 } else { index }];
                let rhs = right[if broadcast[1] { 0 } else { index }];
                let (expected, diagnostic) = expected(lhs, rhs, operation, range);
                assert_eq!(
                    value,
                    expected.encode_le().as_ref(),
                    "{:?}/{operation}/{range:?}/{broadcast:?} lane{index}",
                    T::TYPE
                );
                assert_eq!(
                    status,
                    diagnostic,
                    "{:?}/{operation}/{range:?}/{broadcast:?} lane{index}",
                    T::TYPE
                );
            }
        }
    }
}
fn edges<T: Sample>() -> Vec<T> {
    let width = T::ENCODED_SIZE;
    let mut result = Vec::new();
    for fill in [0, 1, 0x7f, 0x80, 0xff] {
        result.push(T::raw(&vec![fill; width]));
    }
    for index in 0..width {
        let mut bytes = vec![0; width];
        bytes[index] = 1;
        result.push(T::raw(&bytes));
        bytes[index] = 0x80;
        result.push(T::raw(&bytes));
        bytes.fill(0xff);
        bytes[index] = 0;
        result.push(T::raw(&bytes));
        bytes.fill(0);
        bytes[..index].fill(0xff);
        result.push(T::raw(&bytes));
    }
    for sign in [0x7f, 0x80] {
        let mut bytes = vec![if sign == 0x7f { 0xff } else { 0 }; width];
        bytes[width - 1] = sign;
        result.push(T::raw(&bytes));
    }
    result
}
fn pairs<T: Sample>() -> (Vec<T>, Vec<T>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    if T::ENCODED_SIZE == 1 {
        for a in 0..=255 {
            for b in 0..=255 {
                left.push(T::raw(&[a]));
                right.push(T::raw(&[b]));
            }
        }
        return (left, right);
    }
    let edges = edges::<T>();
    for &a in &edges {
        for &b in &edges {
            left.push(a);
            right.push(b);
        }
    }
    let mut state = 0xa076_1d64_78bd_642f_u64;
    while left.len() < 65_536 {
        let mut bytes = vec![0; T::ENCODED_SIZE * 2];
        for byte in &mut bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state.to_le_bytes()[0];
        }
        left.push(T::raw(&bytes[..T::ENCODED_SIZE]));
        right.push(T::raw(&bytes[T::ENCODED_SIZE..]));
    }
    (left, right)
}
fn run<T: Sample>(session: &Session) {
    let (left, right) = pairs::<T>();
    eprintln!(
        "integer bit/status oracle {:?}: direct={} broadcast={}",
        T::TYPE,
        left.len() * 6,
        8 * 2 * 1024 * 6
    );
    qualify(session, &left, &right, [false; 2]);
    for scalar in edges::<T>().into_iter().take(8) {
        qualify(session, &[scalar], &right[..1024], [true, false]);
        qualify(session, &left[..1024], &[scalar], [false, true]);
    }
}
#[test]
#[ignore = "Requires actual independently emitted fourteen-width MLX integer payload/status kernels."]
fn native_fourteen_integer_width_bit_and_fault_oracle() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let native = session.native_for_binary_proof();
    macro_rules! all {($($ty:ty),+)=>{$(run::<$ty>(native);)+};}
    all!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}
#[test]
fn range_status_classes_and_fatal_precedence() {
    assert_eq!(
        select_fault(&[0x101, 3, 1]).unwrap().unwrap(),
        PcuExecutionFault {
            invocation_id: 1,
            kind: Kind::ArithmeticUnderflow,
            recovered: false,
        }
    );
    assert_eq!(
        select_fault(&[0, 0x103, 0x101])
            .unwrap()
            .unwrap()
            .invocation_id,
        1
    );
    assert!(select_fault(&[2]).is_err());
    assert!(select_fault(&[0x102]).is_err());
    assert!(select_fault(&[4]).is_err());
}

#[test]
#[ignore = "Requires actual MLX integer private output recovery, session affinity and retry."]
fn native_integer_private_recovery_and_preflight() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let native = session.native_for_binary_proof();
    let scalar = PcuScalarType::U512;
    let maximum = [0xff_u8; 64];
    let mut one = [0_u8; 64];
    one[0] = 1;
    let left = EncodedArray::upload(native, scalar, 1, &maximum).unwrap();
    let right = EncodedArray::upload(native, scalar, 1, &one).unwrap();
    let wrong = EncodedArray::upload(foreign.native_for_binary_proof(), scalar, 1, &one).unwrap();
    for range in [Range::Reject, Range::Clamp] {
        let prepared =
            CheckedInteger::prepare(native, scalar, 0, range, 1, [1, 1], [false; 2]).unwrap();
        assert!(matches!(
            prepared.execute_encoded([&left, &wrong]),
            Err(MlxError::ForeignSession)
        ));
        assert!(!prepared.may_have_written.get());
        let outcome = prepared.execute_encoded([&left, &right]);
        assert!(prepared.may_have_written.get());
        if range == Range::Reject {
            assert!(matches!(
                outcome,
                Err(MlxError::Arithmetic(PcuExecutionFault {
                    invocation_id: 0,
                    kind: Kind::ArithmeticOverflow,
                    recovered: false,
                }))
            ));
        } else {
            let (output, fault) = outcome.unwrap();
            assert_eq!(
                fault,
                Some(PcuExecutionFault {
                    invocation_id: 0,
                    kind: Kind::ArithmeticOverflow,
                    recovered: true,
                })
            );
            let mut bytes = [0; 64];
            output.read(&mut bytes).unwrap();
            assert_eq!(bytes, maximum);
        }
        // The completed private error/result never mutates either prior immutable owner.
        let mut bytes = [0; 64];
        left.read(&mut bytes).unwrap();
        assert_eq!(bytes, maximum);
        right.read(&mut bytes).unwrap();
        assert_eq!(bytes, one);
        let zero = EncodedArray::upload(native, scalar, 1, &[0; 64]).unwrap();
        let (output, fault) = prepared.execute_encoded([&zero, &right]).unwrap();
        assert!(fault.is_none());
        output.read(&mut bytes).unwrap();
        assert_eq!(bytes, one);
    }
}

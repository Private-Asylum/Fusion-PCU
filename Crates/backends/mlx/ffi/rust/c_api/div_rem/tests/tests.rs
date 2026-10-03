//! Independent exact checked integer quotient/remainder oracle and joint publication.
#[path = "wide/wide.rs"]
mod wide;
#[rustfmt::skip]
use super::{
    CheckedDivRem,
    select_fault,
    EncodedArray,
    Session,
    MlxError,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedIntegerDivision,
    PcuScalar,
    PcuHostArgument,
    PcuBindingRef,
};
trait Sample: PcuCheckedIntegerDivision {
    fn raw(word: u64) -> Self;
    fn zero() -> Self {
        Self::raw(0)
    }
}
macro_rules! sample {
    ($($ty:ty=>$width:literal;)+) => {$(impl Sample for $ty {
        fn raw(word:u64)->Self {Self::decode_le(word.to_le_bytes()[..$width].try_into().unwrap())}
    })+};
}
sample! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;}
fn expected<T: Sample>(left: T, right: T) -> (T, T, u32) {
    match (left.pcu_checked_div(right), left.pcu_checked_rem(right)) {
        (Ok(q), Ok(r)) => (q, r, 0),
        (Err(a), Err(b)) => {
            assert_eq!(a, b);
            (
                T::zero(),
                T::zero(),
                match a {
                    PcuExecutionFaultKind::DivideByZero => 4,
                    PcuExecutionFaultKind::SignedDivisionOverflow => 1,
                    other => panic!("unexpected exact division fault {other:?}"),
                },
            )
        }
        _ => panic!("joint divide/remainder oracle disagreed"),
    }
}
fn qualify<T: Sample>(session: &Session, left: &[T], right: &[T], broadcast: [bool; 2]) {
    let count = left.len().max(right.len());
    let a = PcuHostArgument::read(PcuBindingRef::new(0, 0), left);
    let b = PcuHostArgument::read(PcuBindingRef::new(0, 1), right);
    let a = EncodedArray::upload(session, T::TYPE, left.len(), a.bytes()).unwrap();
    let b = EncodedArray::upload(session, T::TYPE, right.len(), b.bytes()).unwrap();
    let prepared = CheckedDivRem::prepare(
        session,
        T::TYPE,
        count,
        [left.len(), right.len()],
        broadcast,
    )
    .unwrap();
    let ([q, r], records) = prepared.execute_recorded([&a, &b]).unwrap();
    let statuses = prepared.status_records(&records).unwrap();
    let mut qb = vec![0; count * T::ENCODED_SIZE];
    let mut rb = vec![0; count * T::ENCODED_SIZE];
    q.read(&mut qb).unwrap();
    r.read(&mut rb).unwrap();
    for index in 0..count {
        let (want_q, want_r, status) = expected(
            left[if broadcast[0] { 0 } else { index }],
            right[if broadcast[1] { 0 } else { index }],
        );
        let span = index * T::ENCODED_SIZE..(index + 1) * T::ENCODED_SIZE;
        assert_eq!(
            &qb[span.clone()],
            want_q.encode_le().as_ref(),
            "quotient {:?} lane{index}",
            T::TYPE
        );
        assert_eq!(
            &rb[span],
            want_r.encode_le().as_ref(),
            "remainder {:?} lane{index}",
            T::TYPE
        );
        assert_eq!(statuses[index], status, "status {:?} lane{index}", T::TYPE);
    }
    eprintln!(
        "div/rem bit/status oracle {:?}: count={count} broadcast={broadcast:?}",
        T::TYPE
    );
}
fn run<T: Sample>(session: &Session) {
    let width = T::TYPE.bit_width();
    let mut left = Vec::with_capacity(65536);
    let mut right = Vec::with_capacity(65536);
    if width == 8 {
        for a in 0..256 {
            for b in 0..256 {
                left.push(T::raw(a));
                right.push(T::raw(b));
            }
        }
    } else {
        let high = 1_u64 << (width - 1);
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1_u64 << width) - 1
        };
        let edges = [0, 1, 2, 3, high - 1, high, high + 1, mask - 1, mask];
        for a in edges {
            for b in edges {
                left.push(T::raw(a));
                right.push(T::raw(b));
            }
        }
        let mut seed = 0x243f_6a88_85a3_08d3_u64;
        while left.len() < 65536 {
            seed = seed.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            left.push(T::raw(seed));
            seed = seed.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            right.push(T::raw(seed));
        }
    }
    qualify(session, &left, &right, [false; 2]);
    for scalar in [
        T::raw(0),
        T::raw(1),
        T::raw(u64::MAX),
        T::raw(1_u64 << (width - 1)),
    ] {
        qualify(session, &[scalar], &right[..1024], [true, false]);
        qualify(session, &left[..1024], &[scalar], [false, true]);
    }
}
#[test]
#[ignore = "Requires actual independent MLX U32 quotient/remainder/status siblings on Apple GPU."]
fn native_eight_width_div_rem_bit_and_fault_oracle() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let native = session.native_for_binary_proof();
    run::<u8>(native);
    run::<i8>(native);
    run::<u16>(native);
    run::<i16>(native);
    run::<u32>(native);
    run::<i32>(native);
    run::<u64>(native);
    run::<i64>(native);
}
#[test]
fn exact_div_rem_fatal_classes_and_first_logical_lane() {
    assert_eq!(
        select_fault(&[0, 4, 1]).unwrap(),
        Some(PcuExecutionFault {
            invocation_id: 1,
            kind: PcuExecutionFaultKind::DivideByZero,
            recovered: false
        })
    );
    assert_eq!(
        select_fault(&[0, 1, 4]).unwrap(),
        Some(PcuExecutionFault {
            invocation_id: 1,
            kind: PcuExecutionFaultKind::SignedDivisionOverflow,
            recovered: false
        })
    );
    assert_eq!(select_fault(&[0, 0]).unwrap(), None);
    for invalid in [2, 3, 0x101, 0x104] {
        assert!(select_fault(&[invalid]).is_err());
    }
}
#[test]
#[ignore = "Requires actual MLX joint division preflight, immutable owners and terminal host publication."]
fn native_div_rem_joint_publication_and_session_preflight() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let native = session.native_for_binary_proof();
    let input = [i64::MIN, 17, -17, 3, 0];
    let divisor = [1_i64, 5, 5, -1, 3];
    let a = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input);
    let b = PcuHostArgument::read(PcuBindingRef::new(0, 1), &divisor);
    let left = EncodedArray::upload(native, i64::TYPE, 5, a.bytes()).unwrap();
    let right = EncodedArray::upload(native, i64::TYPE, 5, b.bytes()).unwrap();
    let other =
        EncodedArray::upload(foreign.native_for_binary_proof(), i64::TYPE, 5, b.bytes()).unwrap();
    let mut prepared = CheckedDivRem::prepare(native, i64::TYPE, 5, [5; 2], [false; 2]).unwrap();
    assert!(matches!(
        prepared.execute_encoded([&left, &other]),
        Err(MlxError::ForeignSession)
    ));
    assert!(!prepared.may_have_written.get());
    let [q, r] = prepared.execute_encoded([&left, &right]).unwrap();
    let mut out_q = [0xa7_u8; 56];
    let mut out_r = [0xb9_u8; 56];
    prepared
        .execute_host([a.bytes(), b.bytes()], [&mut out_q, &mut out_r])
        .unwrap();
    let expected_q = [i64::MIN, 3, -3, -3, 0];
    let expected_r = [0_i64, 2, -2, 0, 0];
    assert_eq!(
        &out_q[..40],
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &expected_q).bytes()
    );
    assert_eq!(
        &out_r[..40],
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &expected_r).bytes()
    );
    assert_eq!(&out_q[40..], &[0xa7; 16]);
    assert_eq!(&out_r[40..], &[0xb9; 16]);
    let saved_q = out_q;
    let saved_r = out_r;
    assert!(
        prepared
            .execute_host([a.bytes(), b.bytes()], [&mut out_q, &mut out_r[..39]])
            .is_err()
    );
    assert_eq!(out_q, saved_q);
    assert_eq!(out_r, saved_r);
    assert!(!prepared.may_have_written.get());
    let bad = [-1_i64, 5, 0, -1, 3];
    let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), &bad);
    assert_eq!(
        prepared
            .execute_host([a.bytes(), bytes.bytes()], [&mut out_q, &mut out_r])
            .err(),
        Some(MlxError::Arithmetic(PcuExecutionFault {
            invocation_id: 0,
            kind: PcuExecutionFaultKind::SignedDivisionOverflow,
            recovered: false
        }))
    );
    assert_eq!(out_q, saved_q);
    assert_eq!(out_r, saved_r);
    prepared
        .execute_host([a.bytes(), b.bytes()], [&mut out_q, &mut out_r])
        .unwrap();
    drop(prepared);
    drop(left);
    drop(right);
    let mut bytes_q = [0; 40];
    let mut bytes_r = [0; 40];
    q.read(&mut bytes_q).unwrap();
    r.read(&mut bytes_r).unwrap();
    assert_eq!(bytes_q, &saved_q[..40]);
    assert_eq!(bytes_r, &saved_r[..40]);
}

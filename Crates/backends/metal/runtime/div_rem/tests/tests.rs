//! Exact private packed GPU payload/status oracle; only eight-bit pair domains are exhaustive.
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuCheckedIntegerDivision,
    PcuExecutionFaultKind,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
use crate::MetalSession;
trait Sample: PcuCheckedIntegerDivision {
    fn raw(seed: u64) -> Self;
    fn byte(value: u8) -> Self;
}
macro_rules! samples{($($ty:ty=>$width:literal;)+)=>{$(
 impl Sample for $ty{
  fn byte(value:u8)->Self{let mut bytes=[0;$width];bytes[0]=value;Self::decode_le(bytes)}
  fn raw(mut seed:u64)->Self{let phase=seed;let mut bytes=[0;$width];
   for (index,byte)in bytes.iter_mut().enumerate(){seed=seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    *byte=match phase{0=>0,1=>u8::MAX,2=>if index+1==$width{0x80}else{0},3=>if index+1==$width{0x7f}else{u8::MAX},4=>if index==0{1}else{0},_=>seed.to_le_bytes()[7]};}
   Self::decode_le(bytes)
  }
 }
)+};}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;}
samples! {u128=>16;i128=>16;PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
fn prove<T: Sample>(session: &MetalSession, left: &[T], right: &[T]) {
    let encode = |values: &[T]| {
        values
            .iter()
            .flat_map(|v| v.encode_le().as_ref().to_vec())
            .collect::<Vec<_>>()
    };
    let a = session.upload_bytes(&encode(left)).unwrap();
    let b = session.upload_bytes(&encode(right)).unwrap();
    let count = left.len();
    assert_eq!(count, right.len());
    let bytes = count * T::HOST_SIZE;
    let control = if T::TYPE.bit_width() > 64 {
        session.prepare_wide_div_rem_proof(T::TYPE, count)
    } else {
        session.prepare_checked_div_rem_control(T::TYPE, count)
    }
    .unwrap();
    let output = session.allocate_zeroed_bytes(bytes * 2).unwrap();
    let records = session.0.native.allocate(count * 4).unwrap();
    records.fill_ones();
    session
        .0
        .native
        .execute(
            &control.pipeline,
            [&a.native, &b.native, &output.native, &records],
            [u32::try_from(count).unwrap(), 0],
            count,
        )
        .unwrap();
    let mut actual = vec![0; bytes * 2];
    output.read_into_bytes(&mut actual).unwrap();
    records
        .inspect_words(count, |status| {
            for (index, ((&a, &b), &record)) in left.iter().zip(right).zip(status).enumerate() {
                let expected = a
                    .pcu_checked_div(b)
                    .and_then(|q| a.pcu_checked_rem(b).map(|r| (q, r)));
                let code = expected.as_ref().map_or_else(
                    |kind| match kind {
                        PcuExecutionFaultKind::DivideByZero => 2,
                        PcuExecutionFaultKind::SignedDivisionOverflow => 5,
                        _ => panic!("invalid reference DivRem kind"),
                    },
                    |_| 0,
                );
                assert_eq!(record, code, "type={:?},lane={index}", T::TYPE);
                let (q, r) = expected.unwrap_or_else(|_| (T::raw(0), T::raw(0)));
                assert_eq!(
                    &actual[index * T::HOST_SIZE..(index + 1) * T::HOST_SIZE],
                    q.encode_le().as_ref(),
                    "quotient {:?} lane{index}",
                    T::TYPE
                );
                assert_eq!(
                    &actual[bytes + index * T::HOST_SIZE..bytes + (index + 1) * T::HOST_SIZE],
                    r.encode_le().as_ref(),
                    "remainder {:?} lane{index}",
                    T::TYPE
                );
            }
            Ok(())
        })
        .unwrap();
    eprintln!(
        "Metal private DivRem paired bit/status oracle {:?}: count={count}",
        T::TYPE
    );
}
fn wide<T: Sample>(session: &MetalSession) {
    let left: Vec<T> = (0..8192)
        .map(|i| T::raw(u64::try_from(if i < 25 { i / 5 } else { i }).unwrap()))
        .collect();
    let right: Vec<T> = (0..8192)
        .map(|i| {
            T::raw(
                u64::try_from(if i < 25 || i % 4 == 0 {
                    i % 5
                } else {
                    i * 71 + 23
                })
                .unwrap(),
            )
        })
        .collect();
    prove(session, &left, &right);
    let a = [T::raw(2), T::raw(4), T::raw(3)];
    let b = [T::raw(4), T::raw(4), T::raw(4)];
    let sentinel = T::byte(19);
    let mut quotient = [sentinel; 5];
    let mut remainder = [sentinel; 5];
    let mut control = session.prepare_wide_div_rem_proof(T::TYPE, 3).unwrap();
    control.call(&a, &b, &mut quotient, &mut remainder).unwrap();
    for lane in 0..3 {
        assert_eq!(
            quotient[lane].encode_le().as_ref(),
            a[lane]
                .pcu_checked_div(b[lane])
                .unwrap()
                .encode_le()
                .as_ref()
        );
        assert_eq!(
            remainder[lane].encode_le().as_ref(),
            a[lane]
                .pcu_checked_rem(b[lane])
                .unwrap()
                .encode_le()
                .as_ref()
        );
    }
    let before_q = quotient;
    let before_r = remainder;
    assert!(
        control
            .call(&a, &[T::raw(0); 3], &mut quotient, &mut remainder)
            .is_err()
    );
    for lane in 0..5 {
        assert_eq!(
            quotient[lane].encode_le().as_ref(),
            before_q[lane].encode_le().as_ref()
        );
        assert_eq!(
            remainder[lane].encode_le().as_ref(),
            before_r[lane].encode_le().as_ref()
        );
    }
    if matches!(
        a[0].pcu_checked_div(T::raw(1)),
        Err(PcuExecutionFaultKind::SignedDivisionOverflow)
    ) {
        let crate::MetalError::Arithmetic(fault) = control
            .call(
                &a,
                &[T::raw(1), T::raw(4), T::raw(0)],
                &mut quotient,
                &mut remainder,
            )
            .unwrap_err()
        else {
            panic!("missing exact signed wide division fault");
        };
        assert_eq!(fault.kind, PcuExecutionFaultKind::SignedDivisionOverflow);
        assert_eq!(fault.invocation_id, 0);
        assert!(!fault.recovered);
        for lane in 0..5 {
            assert_eq!(
                quotient[lane].encode_le().as_ref(),
                before_q[lane].encode_le().as_ref()
            );
            assert_eq!(
                remainder[lane].encode_le().as_ref(),
                before_r[lane].encode_le().as_ref()
            );
        }
    }
    control.call(&a, &b, &mut quotient, &mut remainder).unwrap();
    for lane in 3..5 {
        assert_eq!(
            quotient[lane].encode_le().as_ref(),
            sentinel.encode_le().as_ref()
        );
        assert_eq!(
            remainder[lane].encode_le().as_ref(),
            sentinel.encode_le().as_ref()
        );
    }
}
#[test]
#[ignore = "Requires actual private wide Metal U32 quotient/remainder/status synthesis and joint rollback."]
fn six_wide_integer_div_rem_private_oracle_and_joint_publication() {
    let session = MetalSession::open(0).unwrap();
    wide::<u128>(&session);
    wide::<i128>(&session);
    wide::<PcuU256>(&session);
    wide::<PcuI256>(&session);
    wide::<PcuU512>(&session);
    wide::<PcuI512>(&session);
}
#[test]
fn div_rem_fault_classes_and_earliest_invocation_remain_distinct() {
    use super::select_div_rem_fault;
    let crate::MetalError::Arithmetic(fault) = select_div_rem_fault(&[0, 5, 2]).unwrap_err() else {
        panic!("missing signed division fault")
    };
    assert_eq!(fault.kind, PcuExecutionFaultKind::SignedDivisionOverflow);
    assert_eq!(fault.invocation_id, 1);
    assert!(!fault.recovered);
    let crate::MetalError::Arithmetic(fault) = select_div_rem_fault(&[2, 5]).unwrap_err() else {
        panic!("missing zero divisor")
    };
    assert_eq!(fault.kind, PcuExecutionFaultKind::DivideByZero);
    assert_eq!(fault.invocation_id, 0);
    assert!(matches!(
        select_div_rem_fault(&[u32::MAX]),
        Err(crate::MetalError::Runtime(_))
    ));
}
#[test]
#[ignore = "Requires actual primitive eight-width packed DivRem U32 GPU oracle."]
fn eight_width_complete_small_and_wide_div_rem_oracle() {
    let session = MetalSession::open(0).unwrap();
    macro_rules! small{($($ty:ty),+)=>{$(
  let a=(0..65_536).map(|i|<$ty>::byte(u8::try_from(i>>8).unwrap())).collect::<Vec<_>>();
  let b=(0..65_536).map(|i|<$ty>::byte(u8::try_from(i&255).unwrap())).collect::<Vec<_>>();prove(&session,&a,&b);
 )+};}
    small!(u8, i8);
    macro_rules! large{($($ty:ty),+)=>{$(
  let a=(0..65_536).map(|i|<$ty>::raw(u64::try_from(i).unwrap())).collect::<Vec<_>>();
  let b=(0..65_536).map(|i|<$ty>::raw(u64::try_from(if i%4==0{i%16}else{i*71+23}).unwrap())).collect::<Vec<_>>();prove(&session,&a,&b);
 )+};}
    large!(u16, i16, u32, i32, u64, i64);
}

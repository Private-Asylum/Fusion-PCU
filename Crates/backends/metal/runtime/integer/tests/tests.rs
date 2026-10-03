//! Native private payload/status proof against the independently sealed checked/clamped reference.
#[rustfmt::skip]
use fusion_pcu::{PcuScalar,PcuCheckedInteger,PcuRangePolicy,PcuExecutionFaultKind,PcuU256,PcuI256,PcuU512,PcuI512};
#[rustfmt::skip]
use super::super::{MetalSession,MetalIntegerOp};
trait Sample: PcuCheckedInteger {
    fn raw(seed: u64) -> Self;
    fn byte(value: u8) -> Self;
    fn half(seed: u64) -> Self;
}
macro_rules! samples {($($ty:ty=>$width:literal;)+)=>{$(
    impl Sample for $ty {
        fn raw(mut seed:u64)->Self {
            let phase=seed;
            let mut bytes=[0_u8;$width];
            for (index,byte) in bytes.iter_mut().enumerate(){
                seed=seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                *byte=match phase {
                    0=>0,1=>u8::MAX,2=>if index+1==$width {0x80}else{0},
                    3=>if index+1==$width {0x7f}else{u8::MAX},
                    4=>if index==0 {1}else{0},
                    _=>seed.to_le_bytes()[7],
                };
            }
            Self::decode_le(bytes)
        }
        fn byte(value:u8)->Self {let mut bytes=[0_u8;$width];bytes[0]=value;Self::decode_le(bytes)}
        fn half(seed:u64)->Self {let mut bytes=Self::raw(seed).encode_le();bytes[$width/2..].fill(0);Self::decode_le(bytes)}
    }
)+};}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;
PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
fn code(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 1,
        PcuExecutionFaultKind::ArithmeticUnderflow => 3,
        _ => panic!("integer oracle emitted non-range fault"),
    }
}
fn checked<T: Sample>(a: T, b: T, op: MetalIntegerOp) -> Result<T, PcuExecutionFaultKind> {
    match op {
        MetalIntegerOp::Add => a.pcu_checked_add(b),
        MetalIntegerOp::Subtract => a.pcu_checked_sub(b),
        MetalIntegerOp::Multiply => a.pcu_checked_mul(b),
        _ => unreachable!("three binary operations"),
    }
}
fn clamp<T: Sample>(a: T, b: T, op: MetalIntegerOp) -> T {
    match op {
        MetalIntegerOp::Add => a.pcu_clamped_add(b),
        MetalIntegerOp::Subtract => a.pcu_clamped_sub(b),
        MetalIntegerOp::Multiply => a.pcu_clamped_mul(b),
        _ => unreachable!("three binary operations"),
    }
    .unwrap_or_else(fusion_pcu::PcuClampedFault::clamped_value)
}
fn prove<T: Sample>(session: &MetalSession, left: &[T], right: &[T]) {
    assert_eq!(left.len(), right.len());
    let encode = |values: &[T]| {
        values
            .iter()
            .flat_map(|value| value.encode_le().as_ref().to_vec())
            .collect::<Vec<_>>()
    };
    let a = session.upload_bytes(&encode(left)).unwrap();
    let b = session.upload_bytes(&encode(right)).unwrap();
    let count = left.len();
    let bytes = count * T::HOST_SIZE;
    for op in [
        MetalIntegerOp::Add,
        MetalIntegerOp::Subtract,
        MetalIntegerOp::Multiply,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let map = session
                .prepare_checked_integer_map(T::TYPE, op, range, count, [false; 2])
                .unwrap();
            let output = session.allocate_zeroed_bytes(bytes).unwrap();
            let records = session.0.native.allocate(count * 4).unwrap();
            records.fill_ones();
            session
                .0
                .native
                .execute(
                    &map.pipeline,
                    [&a.native, &b.native, &output.native, &records],
                    [u32::try_from(count).unwrap(), map.operation],
                    count,
                )
                .unwrap();
            let mut actual = vec![0_u8; bytes];
            output.read_into_bytes(&mut actual).unwrap();
            records
                .inspect_words(count, |status| {
                    for (index, ((&lhs, &rhs), &record)) in
                        left.iter().zip(right).zip(status).enumerate()
                    {
                        let result = checked(lhs, rhs, op);
                        let expected_code = result.as_ref().map_or_else(
                            |&fault| {
                                code(fault)
                                    | if range == PcuRangePolicy::Clamp {
                                        0x100
                                    } else {
                                        0
                                    }
                            },
                            |_| 0,
                        );
                        assert_eq!(
                            record,
                            expected_code,
                            "type={:?},op={op:?},range={range:?},lane={index}",
                            T::TYPE
                        );
                        let expected = match result {
                            Ok(value) => value,
                            Err(_) if range == PcuRangePolicy::Clamp => clamp(lhs, rhs, op),
                            Err(_) => T::raw(0),
                        };
                        assert_eq!(
                            &actual[index * T::HOST_SIZE..(index + 1) * T::HOST_SIZE],
                            expected.encode_le().as_ref(),
                            "type={:?},op={op:?},range={range:?},lane={index}",
                            T::TYPE
                        );
                    }
                    Ok(())
                })
                .unwrap();
        }
    }
}
fn corpus<T: Sample>(session: &MetalSession) {
    let count = 65_536;
    // Half-width products keep meaningful full-width payloads in range, rather than
    // letting randomized full-width multiplication prove only overflow detection.
    let left = (0..count)
        .map(|index| {
            let seed = u64::try_from(index).unwrap();
            if index % 4 == 1 {
                T::half(seed)
            } else {
                T::raw(seed)
            }
        })
        .collect::<Vec<_>>();
    let right = (0..count)
        .map(|index| {
            let seed = u64::try_from(index).unwrap();
            match index % 4 {
                1 => T::half(seed.wrapping_mul(37).wrapping_add(19)),
                2 => T::byte(u8::try_from(index % 3 + 1).unwrap()),
                3 => T::raw(seed.wrapping_mul(71).wrapping_add(23)),
                _ => T::raw(u64::try_from(index % 16).unwrap()),
            }
        })
        .collect::<Vec<_>>();
    prove(session, &left, &right);
}
#[test]
#[ignore = "Requires actual Metal fourteen-width checked integer payload/status oracle."]
fn fourteen_width_complete_small_and_wide_integer_oracle() {
    let session = MetalSession::open(0).unwrap();
    macro_rules! small {($($ty:ty),+)=>{$(
        let a=(0..65_536).map(|index|<$ty>::byte(u8::try_from(index>>8).unwrap())).collect::<Vec<_>>();
        let b=(0..65_536).map(|index|<$ty>::byte(u8::try_from(index&255).unwrap())).collect::<Vec<_>>();
        prove(&session,&a,&b);
    )+};}
    small!(u8, i8);
    macro_rules! large {($($ty:ty),+)=>{$(corpus::<$ty>(&session);)+};}
    large!(
        u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

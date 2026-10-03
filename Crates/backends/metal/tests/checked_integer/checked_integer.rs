//! Fourteen exact checked integer source widths, rollback, observable Clamp and host tails.
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{PcuCheckedInteger,PcuScalar,PcuHostDispatchError,PcuExecutionFaultKind,
    PcuU256,PcuI256,PcuU512,PcuI512};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalError,MetalHostKernelError};
trait Sample: PcuCheckedInteger + std::fmt::Debug {
    fn raw(phase: u8) -> Self;
}
macro_rules! samples {($($ty:ty=>$width:literal;)+)=>{$(
    impl Sample for $ty {fn raw(phase:u8)->Self {
        let mut bytes=[phase;$width];
        match phase {0=>bytes.fill(0),1=>bytes.fill(u8::MAX),2=>{bytes.fill(0);bytes[$width-1]=0x80;},
            3=>{bytes.fill(u8::MAX);bytes[$width-1]=0x7f;},4=>{bytes.fill(0);bytes[0]=1;},_=>{}}
        Self::decode_le(bytes)
    }}
)+};}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;
PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
fn same<T: PcuScalar>(left: &[T], right: &[T]) {
    for (a, b) in left.iter().zip(right) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn pairs<T: Sample>(
    call: &mut impl FnMut(&[T], &[T], &mut [T]) -> Result<(), MetalHostKernelError>,
    op: usize,
    clamp: bool,
) {
    let sentinel = T::raw(19);
    let mut output = [sentinel; 3];
    for a in 0..16 {
        for b in 0..16 {
            let left = T::raw(a);
            let right = T::raw(b);
            let before = output;
            let expected = match op {
                0 => left.pcu_checked_add(right),
                1 => left.pcu_checked_sub(right),
                2 => left.pcu_checked_mul(right),
                _ => unreachable!(),
            };
            let result = call(&[left], &[right], &mut output);
            match expected {
                Ok(value) => {
                    result.unwrap();
                    same(&output, &[value, sentinel, sentinel]);
                }
                Err(kind) => {
                    let Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) = result
                    else {
                        panic!("missing checked range fault");
                    };
                    assert_eq!(fault.kind, kind);
                    assert_eq!(fault.invocation_id, 0);
                    assert_eq!(fault.recovered, clamp);
                    if clamp {
                        let expected = match op {
                            0 => left.pcu_clamped_add(right),
                            1 => left.pcu_clamped_sub(right),
                            2 => left.pcu_clamped_mul(right),
                            _ => unreachable!(),
                        }
                        .unwrap_err()
                        .clamped_value();
                        same(&output, &[expected, sentinel, sentinel]);
                    } else {
                        same(&output, &before);
                    }
                }
            }
        }
    }
    let before = output;
    assert!(call(&[], &[T::raw(4)], &mut output).is_err());
    same(&output, &before);
    // Retry must be valid for unsigned Sub too: zero-minus-one is a real fault.
    let right = if op == 1 { T::raw(0) } else { T::raw(4) };
    call(&[T::raw(0)], &[right], &mut output).unwrap();
    let expected = if op == 0 { T::raw(4) } else { T::raw(0) };
    same(&output[..1], &[expected]);
    same(&output[1..], &[sentinel; 2]);
}
fn qualify<T: Sample>(session: &MetalSession) {
    pairs(
        &mut source::add_prepare::<T, 1, _>(session).unwrap(),
        0,
        false,
    );
    pairs(
        &mut source::sub_prepare::<T, 1, _>(session).unwrap(),
        1,
        false,
    );
    pairs(
        &mut source::mul_prepare::<T, 1, _>(session).unwrap(),
        2,
        false,
    );
    pairs(
        &mut source::add_clamp_prepare::<T, 1, _>(session).unwrap(),
        0,
        true,
    );
    pairs(
        &mut source::sub_clamp_prepare::<T, 1, _>(session).unwrap(),
        1,
        true,
    );
    pairs(
        &mut source::mul_clamp_prepare::<T, 1, _>(session).unwrap(),
        2,
        true,
    );
    let a = [T::raw(0); 65];
    let b = [T::raw(4); 65];
    let sentinel = T::raw(19);
    let mut output = [sentinel; 68];
    source::grid_add_prepare::<T, 65, _>(session).unwrap()(&a, &b, &mut output).unwrap();
    same(&output[..65], &b);
    same(&output[65..], &[sentinel; 3]);
    source::scalar_mul_prepare::<T, 65, _>(session).unwrap()(&b, &T::raw(4), &mut output).unwrap();
    same(&output[..65], &b);
    same(&output[65..], &[sentinel; 3]);
}
#[test]
#[ignore = "Requires actual Metal fourteen-width checked and observable Clamp source realization."]
fn fourteen_width_source_range_and_shape_contract() {
    let session = MetalSession::open(0).unwrap();
    macro_rules! all {($($ty:ty),+)=>{$(qualify::<$ty>(&session);)+};}
    all!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}
#[test]
fn independent_signed_range_oracle_classifies_both_endpoints() {
    assert_eq!(
        i8::MIN.pcu_checked_add(-1),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(
        i8::MAX.pcu_checked_add(1),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
}

#[path = "resident/resident.rs"]
mod resident;

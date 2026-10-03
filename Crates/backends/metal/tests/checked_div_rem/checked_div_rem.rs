//! Genuine eight-primitive source paths with precise faults and joint two-output rollback.
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{PcuCheckedIntegerDivision,PcuScalar,PcuHostDispatchError,PcuU256,PcuI256,PcuU512,PcuI512};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalHostKernelError,MetalError};
trait Sample: PcuCheckedIntegerDivision {
    fn raw(phase: u8) -> Self;
    fn decode(bytes: &[u8]) -> Self;
}
macro_rules! samples{($($ty:ty=>$width:literal;)+)=>{$(
 impl Sample for $ty{fn decode(bytes:&[u8])->Self{Self::decode_le(bytes.try_into().unwrap())}fn raw(phase:u8)->Self{let mut bytes=[phase;$width];match phase{0=>bytes.fill(0),1=>bytes.fill(u8::MAX),2=>{bytes.fill(0);bytes[$width-1]=0x80;},3=>{bytes.fill(u8::MAX);bytes[$width-1]=0x7f;},4=>{bytes.fill(0);bytes[0]=1;},_=>{}}Self::decode_le(bytes)}}
)+};}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
fn same<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn pairs<T: Sample>(
    mut call: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), MetalHostKernelError>,
) {
    let sentinel = T::raw(9);
    let mut q = [sentinel; 3];
    let mut r = q;
    for a in 0..16 {
        for b in 0..16 {
            let a = T::raw(a);
            let b = T::raw(b);
            let before = (q, r);
            let expected = a
                .pcu_checked_div(b)
                .and_then(|q| a.pcu_checked_rem(b).map(|r| (q, r)));
            let result = call(&[a], &[b], &mut q, &mut r);
            match expected {
                Ok((eq, er)) => {
                    result.unwrap();
                    same(&q, &[eq, sentinel, sentinel]);
                    same(&r, &[er, sentinel, sentinel]);
                }
                Err(kind) => {
                    let Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) = result
                    else {
                        panic!("missing precise checked DivRem fault")
                    };
                    assert_eq!(fault.kind, kind);
                    assert_eq!(fault.invocation_id, 0);
                    assert!(!fault.recovered);
                    same(&q, &before.0);
                    same(&r, &before.1);
                }
            }
        }
    }
    let before = (q, r);
    assert!(call(&[T::raw(4)], &[T::raw(4)], &mut q, &mut []).is_err());
    same(&q, &before.0);
    same(&r, &before.1);
    call(&[T::raw(4)], &[T::raw(4)], &mut q, &mut r).unwrap();
    same(&q, &[T::raw(4), sentinel, sentinel]);
    same(&r, &[T::raw(0), sentinel, sentinel]);
}
fn grid<T: Sample>(
    mut call: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), MetalHostKernelError>,
) {
    let sentinel = T::raw(9);
    let left = [T::raw(3); 65];
    let mut right = [T::raw(4); 65];
    let mut q = [sentinel; 68];
    let mut r = q;
    right[64] = T::raw(0);
    let before = (q, r);
    let Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) =
        call(&left, &right, &mut q, &mut r)
    else {
        panic!("missing late zero-divisor")
    };
    assert_eq!(fault.invocation_id, 64);
    assert_eq!(fault.kind, pcu_facade::PcuExecutionFaultKind::DivideByZero);
    same(&q, &before.0);
    same(&r, &before.1);
    right[64] = T::raw(4);
    call(&left, &right, &mut q, &mut r).unwrap();
    same(&q[..65], &left);
    same(&r[..65], &[T::raw(0); 65]);
    same(&q[65..], &[sentinel; 3]);
    same(&r[65..], &[sentinel; 3]);
}
#[test]
#[ignore = "Requires actual eight-width Metal source direct/Strict/grid and joint packed-output publication."]
fn eight_primitive_source_faults_and_joint_publication() {
    let session = MetalSession::open(0).unwrap();
    let backend = session;
    macro_rules! all{($($module:ident:$ty:ty),+)=>{$(
  pairs::<$ty>(source::$module::direct_prepare::<1,_>(&backend).unwrap());
  pairs::<$ty>(source::$module::strict_prepare::<1,_>(&backend).unwrap());
  grid::<$ty>(source::$module::grid_prepare::<65,_>(&backend).unwrap());
 )+};}
    all!(u8_source:u8,i8_source:i8,u16_source:u16,i16_source:i16,u32_source:u32,i32_source:i32,u64_source:u64,i64_source:i64);
}

#[path = "mixed/mixed.rs"]
mod mixed;

#[path = "ordinary/ordinary.rs"]
mod ordinary;
#[path = "roles/roles.rs"]
mod roles;

#[test]
#[ignore = "Requires actual six wide Metal generic source direct/Strict/grid exact faults and joint output rollback."]
fn six_wide_public_source_faults_and_joint_publication() {
    let session = MetalSession::open(0).unwrap();
    let backend = session.checked_div_rem_backend();
    macro_rules! run {($($ty:ty),+) => {$ (
        pairs::<$ty>(source::generic::direct_prepare::<$ty,1,_>(&backend).unwrap());
        pairs::<$ty>(source::generic::strict_prepare::<$ty,1,_>(&backend).unwrap());
        grid::<$ty>(source::generic::grid_prepare::<$ty,65,_>(&backend).unwrap());
    )+};}
    run!(u128, i128, PcuU256, PcuI256, PcuU512, PcuI512);
}

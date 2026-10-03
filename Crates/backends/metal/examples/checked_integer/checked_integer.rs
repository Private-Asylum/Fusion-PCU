//! Real generic source, wide signed transport, and useful completed borrowed Clamp error.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuCheckedInteger,PcuI512,PcuHostDispatchError,PcuScalar};
use fusion_pcu_metal::{MetalSession, MetalError};
#[pcu(invocations=2,crate_path=::pcu_facade)]
fn add<T: PcuCheckedInteger>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=2,flag(strict),flag(clamp_range),crate_path=::pcu_facade)]
fn add_clamp<T: PcuCheckedInteger>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
fn wide(value: i64) -> PcuI512 {
    let mut bytes = [if value < 0 { u8::MAX } else { 0 }; 64];
    bytes[..8].copy_from_slice(&value.to_le_bytes());
    PcuI512::decode_le(bytes)
}
fn main() {
    if !cfg!(target_os = "macos") {
        println!("SKIP: requires actual Metal GPU");
        return;
    }
    let session = MetalSession::open(0).unwrap();
    let left = [wide(3_i64), wide(-7_i64)];
    let right = [wide(2_i64), wide(4_i64)];
    let sentinel = wide(91_i64);
    let mut output = [sentinel; 3];
    add_prepare::<PcuI512, _>(&session).unwrap()(&left, &right, &mut output).unwrap();
    assert_eq!(output, [wide(5_i64), wide(-3_i64), sentinel]);
    let mut narrow = [91_i8; 3];
    let result = add_clamp_prepare::<i8, _>(&session).unwrap()(&[127, 1], &[1, 2], &mut narrow);
    assert!(
        matches!(result,Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(f))) if f.recovered&&f.invocation_id==0)
    );
    assert_eq!(narrow, [127, 3, 91]);
    println!("Wide checked source completed; narrow Clamp={result:?}, output={narrow:?}");
}

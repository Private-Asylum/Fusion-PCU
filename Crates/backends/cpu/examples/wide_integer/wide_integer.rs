//! Wide limbs use the same generic annotated source and explicit CPU provider.
#[rustfmt::skip]
use pcu_facade::{pcu,global,PcuCheckedInteger,PcuI512};
#[pcu(invocations=2,crate_path=::pcu_facade)]
fn add<T: PcuCheckedInteger>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let one = PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]);
    let high = PcuI512::from_limbs_le([u64::MAX, 4, 0, 0, 0, 0, 0, 0]);
    let mut output = [PcuI512::ZERO; 3];
    add::<PcuI512>(&[high, one], &[one; 2], &mut output).unwrap();
    assert_eq!(output[0].to_limbs_le(), [0, 5, 0, 0, 0, 0, 0, 0]);
    let before = output;
    assert!(
        matches!(add::<PcuI512>(&[one,PcuI512::MAX],&[one;2],&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id==1)
    );
    assert_eq!(output, before);
    println!("512-bit carry is exact; later overflow preserved complete output and tail.");
}

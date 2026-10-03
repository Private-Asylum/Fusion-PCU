//! Ordinary generic source returns a useful saturated wide output with an observable error.
#[rustfmt::skip]
use pcu_facade::{global,pcu,PcuCheckedInteger,PcuI512};
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
fn add<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;
    let one = PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]);
    let two = PcuI512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]);
    let three = PcuI512::from_limbs_le([3, 0, 0, 0, 0, 0, 0, 0]);
    let mut output = [one; 5];
    let left = [PcuI512::MAX, two, two];
    let right = [one; 3];
    assert!(
        matches!(add::<PcuI512,3>(&left,&right,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered&&fault.invocation_id==0)
    );
    assert_eq!(output, [PcuI512::MAX, three, three, one, one]);
    let before = output;
    assert!(add::<PcuI512, 3>(&left[..2], &right, &mut output).is_err());
    assert_eq!(output, before);
    add::<PcuI512, 3>(&[two; 3], &right, &mut output)?;
    assert_eq!(output, [three, three, three, one, one]);
    println!(
        "CPU generic I512 Clamp publishes useful output with Err, preserves tails/preflight output, and retries."
    );
    global::clear_thread_cache()?;
    global::use_defaults()?;
    Ok(())
}

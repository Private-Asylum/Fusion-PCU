//! Ordinary scalar calls borrow the original native owners and preserve their untouched tails.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[pcu(crate_path=::pcu_facade)]
fn own<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(invocations=7,crate_path=::pcu_facade)]
fn copy_prefix<T: PcuScalar>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}
fn main() -> Result<(), PcuExecutionError> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })?;
    let input = own(&[1_u8, 2, 3, 4, 5, 6, 7])?;
    let mut output = own(&[91_u8; 10])?;
    global::clear_thread_cache()?;
    copy_prefix(&input, &mut output)?;
    let mut observed = [0_u8; 10];
    output.read_into(&mut observed)?;
    assert_eq!(observed, [1, 2, 3, 4, 5, 6, 7, 91, 91, 91]);
    println!("same-session Vulkan resident prefix: {observed:?}");
    global::clear_thread_cache()?;
    global::use_defaults()?;
    Ok(())
}

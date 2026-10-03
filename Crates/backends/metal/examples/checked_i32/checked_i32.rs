//! Ordinary typed RAM source staging into retained native Metal checked arithmetic.
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalError,
    MetalSession,
};
use pcu_facade::pcu;

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn product<const N: usize>(left: &[i32], right: &[i32], output: &mut [i32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
fn main() {
    if !cfg!(target_os = "macos") {
        println!("SKIP: checked Metal I32 requires macOS hardware");
        return;
    }
    let session = MetalSession::open(0).unwrap();
    let mut source = product_prepare::<3, _>(&session).unwrap();
    drop(session);
    let mut output = [91_i32; 5];
    source(&[i32::MIN, -7, 12], &[1, -3, 4], &mut output).unwrap();
    assert_eq!(output, [i32::MIN, 21, 48, 91, 91]);
    let before = output;
    let Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) =
        source(&[1, i32::MIN, i32::MAX], &[2, -1, 2], &mut output)
    else {
        panic!("missing checked arithmetic fault");
    };
    assert_eq!(fault.invocation_id, 1);
    assert_eq!(output, before);
    source(&[-2, -3, -4], &[5, 6, 7], &mut output).unwrap();
    assert_eq!(output, [-10, -18, -28, 91, 91]);
    println!("Metal retained checked I32 source output: {output:?}; terminal fault: {fault:?}");
}

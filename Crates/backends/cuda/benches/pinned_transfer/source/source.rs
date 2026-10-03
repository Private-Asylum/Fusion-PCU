//! A real host-to-device-to-host identity workload, with no arithmetic interpretation.
#[rustfmt::skip]
use fusion_pcu::pcu;

#[pcu(invocations = N)]
pub fn copy<const N: usize>(input: &[u8], output: &mut [u8]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

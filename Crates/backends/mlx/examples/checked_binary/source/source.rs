//! Genuine checked low-format arithmetic source, specialized once during preparation.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuCheckedFloat};
#[pcu(invocations=3,flag(strict),crate_path=::pcu_facade)]
pub fn add<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}

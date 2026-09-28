use fusion_pcu_macros::pcu;
#[allow(unused_imports)]
use pcu_alias::PcuScalar;

extern crate pcu_alias;

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn arithmetic<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    output[id / C][id % C] = input[id / C][id % C] + input[id / C][id % C];
}

fn main() {}

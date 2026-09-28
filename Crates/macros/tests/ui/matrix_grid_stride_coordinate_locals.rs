use fusion_pcu_macros::pcu;
use pcu_alias::PcuScalar;

extern crate pcu_alias;

#[pcu(invocations: 3, crate_path = ::pcu_alias)]
fn copy<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < R * C {
        let row = id / C;
        let col = id % C;
        output[row][col] = input[row][col];
        id += stride;
    }
}

fn main() {}

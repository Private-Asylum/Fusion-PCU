use fusion_pcu_macros::pcu;

extern crate pcu_alias;

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn copy<const R: usize, const C: usize>(input: &[[f32; C]; R], output: &mut [[f32; C]; R]) {
    let id = pcu::context::global_invocation_id();
    let row = id / (C + 1);
    let col = id % (C + 1);
    output[row][col] = input[row][col];
}

fn main() {}

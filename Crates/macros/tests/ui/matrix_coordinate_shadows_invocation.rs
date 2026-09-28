use fusion_pcu_macros::pcu;

extern crate pcu_alias;

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn copy<const R: usize, const C: usize>(input: &[[f32; C]; R], output: &mut [[f32; C]; R]) {
    let id = pcu::context::global_invocation_id();
    let id = id / C;
    let col = id % C;
    output[id][col] = input[id][col];
}

fn main() {}

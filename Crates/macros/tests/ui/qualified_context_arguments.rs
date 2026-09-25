use fusion_pcu_macros::pcu;

#[pcu(invocations = 8)]
fn kernel(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id(1);
    output[id] = input[id];
}

fn main() {}

use fusion_pcu_macros::pcu;

#[pcu(invocations = 8)]
fn mixed(input: &[f32], output: &mut [u32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation];
}

fn main() {}

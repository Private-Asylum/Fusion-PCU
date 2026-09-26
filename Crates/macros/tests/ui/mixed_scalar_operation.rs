use fusion_pcu_macros::pcu;

#[pcu(invocations = 8)]
fn mixed(input: &[u32], output: &mut [u32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + 1.0;
}

fn main() {}

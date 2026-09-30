use fusion_pcu_macros::pcu;

#[pcu(invocations = 1, flag(strict), flag(non_strict))]
fn conflict(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * 2.0;
}

fn main() {}

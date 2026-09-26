use fusion_pcu_macros::pcu;

extern crate pcu_alias as renamed_pcu;

#[pcu(invocations = 64, crate_path = ::renamed_pcu)]
fn wrapping_map(input: &[u32], rhs: &[u32], output: &mut [u32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation].wrapping_add(rhs[invocation]);
}

fn main() {}

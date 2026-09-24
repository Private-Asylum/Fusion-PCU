use fusion_pcu_macros::pcu_dispatch;

extern crate pcu_alias as renamed_pcu;

struct Context {
    global_invocation_id: usize,
}

#[pcu_dispatch(invocations = 8, crate_path = ::renamed_pcu)]
fn kernel(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * 2.0;
}

fn main() {}

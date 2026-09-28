extern crate pcu_alias as fusion_pcu;

use fusion_pcu_macros::pcu_dispatch;

fn scale(value: f32) -> f32 {
    value * 2.0
}

#[pcu_dispatch(crate_path = ::pcu_alias, invocations = 8)]
fn kernel(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = scale(input[invocation]);
}

fn main() {}

use fusion_pcu_macros::pcu;

extern crate pcu_alias;

#[pcu(invocations = 1, crate_path = ::pcu_alias)]
fn invalid(input: &[f32], factor: &mut f32, output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = input[id] * *factor;
}

fn main() {}

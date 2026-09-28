use fusion_pcu_macros::pcu;

extern crate pcu_alias;

#[pcu(crate_path = ::pcu_alias)]
fn widen(value: f64) -> f64 {
    value * 2.0
}

#[pcu(invocations = 1, crate_path = ::pcu_alias)]
fn invalid(input: &[f32], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = widen(input[id]);
}

fn main() {}

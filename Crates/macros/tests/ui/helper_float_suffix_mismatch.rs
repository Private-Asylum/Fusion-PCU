use fusion_pcu_macros::pcu;

extern crate pcu_alias;

#[pcu(crate_path = ::pcu_alias)]
fn invalid(value: f64) -> f64 {
    value + 1.0f32
}

fn main() {}

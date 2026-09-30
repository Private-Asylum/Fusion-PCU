use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias, flag(strict))]
fn square(value: f32) -> f32 {
    value * value
}

#[pcu(crate_path = ::pcu_alias, flag(non_strict))]
fn sum(left: f32, right: f32) -> f32 {
    left + right
}

fn main() {}

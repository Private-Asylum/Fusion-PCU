use fusion_pcu_macros::pcu;

#[pcu(flag(strict), flag(strict))]
fn duplicate(value: f32) -> f32 {
    value * value
}

fn main() {}

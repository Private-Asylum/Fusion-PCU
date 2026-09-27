use fusion_pcu_macros::pcu_module;

#[pcu_module]
mod kernels {
    #[pcu_fn]
    fn recurse(value: f32) -> f32 {
        recurse(value)
    }
}

fn main() {}

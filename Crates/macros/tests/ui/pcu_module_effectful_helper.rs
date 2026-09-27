use fusion_pcu_macros::pcu_module;

#[pcu_module]
mod kernels {
    #[pcu_fn]
    fn effect(value: f32) -> f32 {
        let adjusted = value + 1.0;
        adjusted
    }
}

fn main() {}

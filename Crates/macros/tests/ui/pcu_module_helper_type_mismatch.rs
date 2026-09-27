use fusion_pcu_macros::pcu_module;

#[pcu_module]
mod kernels {
    #[pcu_fn]
    fn scale(value: f32) -> f32 {
        value * 2.0
    }

    #[pcu(invocations = 8)]
    fn map(input: &[f64], output: &mut [f64]) {
        let invocation = context.global_invocation_id;
        output[invocation] = scale(input[invocation]);
    }
}

fn main() {}

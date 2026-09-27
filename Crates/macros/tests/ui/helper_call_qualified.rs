use fusion_pcu_macros::pcu_dispatch;

mod math {
    pub fn scale(value: f32) -> f32 {
        value * 2.0
    }
}

#[pcu_dispatch(invocations = 8)]
fn kernel(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = math::scale(input[invocation]);
}

fn main() {}

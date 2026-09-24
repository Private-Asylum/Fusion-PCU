use fusion_pcu_macros::pcu_dispatch;

struct Context {
    global_invocation_id: usize,
}

#[pcu_dispatch(invocations = 8)]
fn kernel(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    if invocation > 0 {
        output[invocation] = input[invocation];
    }
}

fn main() {}

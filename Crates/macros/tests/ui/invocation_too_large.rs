use fusion_pcu_macros::pcu_dispatch;

extern crate pcu_alias as fusion_pcu;

#[pcu_dispatch(invocations = R)]
fn kernel<const R: usize>(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation];
}

fn main() {
    let bindings = kernel_bindings();
    let _ = kernel::<{ u32::MAX as usize + 1 }>(&bindings);
}

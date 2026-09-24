use fusion_pcu_macros::pcu_dispatch;

extern crate pcu_alias as fusion_pcu;

#[pcu_dispatch(invocations = R * C)]
fn kernel<const R: usize, const C: usize>(
    input: &[f32],
    output: &mut [f32],
) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation];
}

fn main() {
    let bindings = kernel_bindings();
    let _ = kernel::<{ usize::MAX }, 2>(&bindings);
}

use fusion_pcu_macros::pcu_dispatch;

extern crate pcu_alias as fusion_pcu;

#[pcu_dispatch(invocations = R * C + 1)]
fn kernel<const R: usize, const C: usize>(
    input: &[[f32; C]; R],
    output: &mut [[f32; C]; R],
) {
    let id = context.global_invocation_id;
    output[id / C][id % C] = input[id / C][id % C];
}

fn main() {
    let bindings = kernel_bindings();
    let _ = kernel::<2, 3>(&bindings);
}

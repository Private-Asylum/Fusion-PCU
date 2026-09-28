use fusion_pcu_macros::pcu_dispatch;

extern crate pcu_alias as fusion_pcu;

#[pcu_dispatch(invocations = R * C)]
fn kernel<const R: usize, const C: usize, const D: usize>(
    input: &[[f32; C]; R],
    output: &mut [[f32; D]; R],
) {
    let id = context.global_invocation_id;
    output[id / D][id % D] = input[id / C][id % C];
}

fn main() {}

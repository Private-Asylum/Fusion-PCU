use fusion_pcu_macros::pcu_dispatch;

extern crate pcu_alias as renamed_pcu;

#[pcu_dispatch(invocations = N + 1, crate_path = ::renamed_pcu)]
fn kernel<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}

fn main() {}

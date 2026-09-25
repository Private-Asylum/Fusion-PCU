use fusion_pcu_macros::pcu_dispatch;

#[pcu_dispatch(invocations = N)]
fn kernel<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        if id == 0 {
            break;
        }
        output[id] = input[id];
        id += stride;
    }
}

fn main() {}

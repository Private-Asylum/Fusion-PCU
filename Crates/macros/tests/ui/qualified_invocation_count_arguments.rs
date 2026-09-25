use fusion_pcu_macros::pcu;

#[pcu(invocations = 4)]
fn kernel<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count(1);
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}

fn main() {}

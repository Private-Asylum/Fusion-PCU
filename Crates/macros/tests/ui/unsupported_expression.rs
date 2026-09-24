use fusion_pcu_macros::pcu_dispatch;

#[allow(non_camel_case_types)]
type read_storage<T> = T;
#[allow(non_camel_case_types)]
type write_storage<T> = T;

struct Context {
    global_invocation_id: usize,
}

#[pcu_dispatch(invocations = 8)]
fn kernel(input: read_storage<f32>, output: write_storage<f32>) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation].abs();
}

fn main() {}

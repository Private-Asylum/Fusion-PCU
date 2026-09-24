use fusion_pcu_macros::pcu_dispatch;

#[allow(non_camel_case_types)]
type read_storage<T> = T;
#[allow(non_camel_case_types)]
type write_storage<T> = T;

struct Context {
    thread: usize,
}

#[pcu_dispatch(threads = 8)]
fn kernel(input: read_storage<f32>, output: write_storage<f32>) {
    let thread = context.thread;
    if thread > 0 {
        output[thread] = input[thread];
    }
}

fn main() {}

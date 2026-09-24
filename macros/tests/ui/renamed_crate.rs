use fusion_pcu_macros::pcu_dispatch;

extern crate pcu_alias as renamed_pcu;

#[allow(non_camel_case_types)]
type read_storage<T> = T;
#[allow(non_camel_case_types)]
type write_storage<T> = T;

struct Context {
    thread: usize,
}

#[pcu_dispatch(threads = 8, crate_path = ::renamed_pcu)]
fn kernel(input: read_storage<f32>, output: write_storage<f32>) {
    let thread = context.thread;
    output[thread] = input[thread] * 2.0;
}

fn main() {}

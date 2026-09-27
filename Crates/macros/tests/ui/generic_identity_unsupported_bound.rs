use fusion_pcu_macros::pcu;

extern crate pcu_alias;

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn generic_clone<T: Clone>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}

fn main() {}

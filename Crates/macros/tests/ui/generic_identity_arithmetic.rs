use fusion_pcu_macros::pcu;
extern crate pcu_alias;

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn generic_add<T: pcu_alias::PcuScalar>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] + input[id];
}

fn main() {}

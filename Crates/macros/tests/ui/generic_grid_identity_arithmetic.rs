use fusion_pcu_macros::pcu;
extern crate pcu_alias;

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn invalid<T: pcu_alias::PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[id] + input[id];
        id += stride;
    }
}

fn main() {}

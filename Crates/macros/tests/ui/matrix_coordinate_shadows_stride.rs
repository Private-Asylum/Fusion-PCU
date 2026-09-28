use fusion_pcu_macros::pcu;

extern crate pcu_alias;

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn copy<const R: usize, const C: usize>(input: &[[f32; C]; R], output: &mut [[f32; C]; R]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < R * C {
        let row = id / C;
        let stride = id % C;
        output[row][stride] = input[row][stride];
        id += stride;
    }
}

fn main() {}

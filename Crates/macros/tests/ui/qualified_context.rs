use fusion_pcu_macros::{
    pcu,
    pcu_dispatch,
};

extern crate pcu_alias as renamed_pcu;

#[pcu(invocations = N, crate_path = ::renamed_pcu)]
fn map<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + 1.0;
}

#[pcu_dispatch(invocations = 4, crate_path = ::renamed_pcu)]
fn grid_stride<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id] * 2.0;
        id += stride;
    }
}

fn main() {
    let map_bindings = map_bindings();
    let _map = map::<32>(&map_bindings).expect("qualified global ID lowers");
    let grid_bindings = grid_stride_bindings();
    let _grid = grid_stride::<16>(&grid_bindings).expect("qualified grid-stride context lowers");
}

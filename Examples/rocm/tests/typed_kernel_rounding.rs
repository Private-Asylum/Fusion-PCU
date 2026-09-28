//! A contraction witness for the ordinary Rust arithmetic contract of typed kernels.

use fusion_pcu_macros::pcu;
use fusion_pcu_rocm::RocmDiscovery;

#[path = "../selection.rs"]
mod selection;

#[pcu(invocations = 1)]
fn separate_rounding(left: &[f32], right: &[f32], bias: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id] + bias[id];
}

#[pcu(invocations = 1)]
fn strided_rounding<const N: usize>(left: &[f32], right: &[f32], bias: &[f32], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id] + bias[id];
        id += stride;
    }
}

#[test]
#[ignore = "requires explicitly selected ROCm hardware"]
fn ordinary_rust_operations_do_not_silently_contract() {
    let discovery = RocmDiscovery::new();
    let candidates = selection::ranked_devices(
        &discovery,
        selection::preferred_device().expect("valid device selection"),
        false,
    )
    .expect("discover ROCm device");
    let (backend, selected) =
        selection::open_ranked(&discovery, candidates, 64).expect("open ROCm device");
    let mut run = separate_rounding_prepare(&backend).expect("prepare contraction witness");
    let left = [f32::from_bits(0x3f80_0001)];
    let right = [f32::from_bits(0x3f7f_fffe)];
    let bias = [-1.0_f32];
    let product = left[0] * right[0];
    let expected = product + bias[0];
    assert_eq!(expected.to_bits(), 0);
    assert_ne!(
        left[0].mul_add(right[0], bias[0]).to_bits(),
        expected.to_bits()
    );
    let mut output = [99.0_f32];
    run(&left, &right, &bias, &mut output).expect("execute witness");
    let mut strided = strided_rounding_prepare::<1, _>(&backend).expect("prepare strided witness");
    strided(&left, &right, &bias, &mut output).expect("execute strided witness");
    assert_eq!(output[0].to_bits(), expected.to_bits(), "strided rounding");
    run(&left, &right, &bias, &mut output).expect("repeat direct witness");
    assert_eq!(
        output[0].to_bits(),
        expected.to_bits(),
        "ordinary Rust rounding on {}",
        selected.name
    );
}

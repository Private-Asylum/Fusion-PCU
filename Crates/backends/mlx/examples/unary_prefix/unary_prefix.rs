//! Genuine annotated scalar read over a longer resident owner; cold full shape, no warm view.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
};
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate_prefix<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
fn main() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let input = session
        .upload_encoded(&[1.0_f64, -0.0, -2.0, f64::NAN, f64::INFINITY])
        .unwrap();
    let bindings = negate_prefix_bindings::<f64>();
    let builder = negate_prefix_ir::<f64, 3>(&bindings).unwrap();
    let mut prepared = builder
        .with_ir(|ir| session.prepare_unary_host_kernel_with_input_extents(ir, &[5]))
        .unwrap();
    let (output, notice) = prepared.execute_resident(&input).unwrap().into_parts();
    assert!(notice.is_none());
    drop(input);
    drop(prepared);
    let mut result = [91.0_f64; 5];
    output.read_into(&mut result).unwrap();
    assert_eq!(
        result.map(f64::to_bits),
        [-1.0_f64, 0.0, 2.0, 91.0, 91.0].map(f64::to_bits)
    );
    println!("MLX-owned F64 unary prefix, unread nonfinite suffix, terminal tails: {result:?}");
}

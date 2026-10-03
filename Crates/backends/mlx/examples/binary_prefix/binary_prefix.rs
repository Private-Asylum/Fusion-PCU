//! Repeated mixed-index reads borrow one longer encoded owner without staging an unused input.
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
fn repeated_prefix<T: PcuCheckedFloat, const N: usize>(
    unused: &[T],
    input: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0] / input[id];
}
fn main() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let input = session
        .upload_encoded(&[2.0_f64, 4.0, 8.0, f64::NAN, 0.0])
        .unwrap();
    let bindings = repeated_prefix_bindings::<f64>();
    let mut prepared = repeated_prefix_ir::<f64, 3>(&bindings)
        .unwrap()
        .with_ir(|ir| {
            session
                .checked_binary_backend()
                .prepare_host_kernel_with_input_extents(ir, &[5])
        })
        .unwrap();
    assert_eq!(prepared.input_bindings().len(), 1);
    let (output, notice) = prepared.execute_resident(&[&input]).unwrap().into_parts();
    assert!(notice.is_none());
    drop(input);
    drop(prepared);
    let mut result = [91.0_f64; 5];
    output.read_into(&mut result).unwrap();
    assert_eq!(
        result.map(f64::to_bits),
        [1.0_f64, 0.5, 0.25, 91.0, 91.0].map(f64::to_bits)
    );
    println!(
        "MLX-owned repeated F64 prefix, unread nonfinite/zero suffix and terminal tails: {result:?}"
    );
}

//! Repeated mixed-index reads borrow one longer encoded owner without staging an unused input.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
};
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedInteger,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn repeated_prefix<T: PcuCheckedInteger, const N: usize>(
    unused: &[T],
    input: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0] - input[id];
}
fn main() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let input = session
        .upload_encoded(&[8_i128, 4, 2, i128::MAX, i128::MIN])
        .unwrap();
    let bindings = repeated_prefix_bindings::<i128>();
    let mut prepared = repeated_prefix_ir::<i128, 3>(&bindings)
        .unwrap()
        .with_ir(|ir| {
            session
                .checked_integer_backend()
                .prepare_host_kernel_with_input_extents(ir, &[5])
        })
        .unwrap();
    assert_eq!(prepared.input_bindings().len(), 1);
    let (output, notice) = prepared.execute_resident(&[&input]).unwrap().into_parts();
    assert!(notice.is_none());
    drop(input);
    drop(prepared);
    let mut result = [91_i128; 5];
    output.read_into(&mut result).unwrap();
    assert_eq!(result, [0_i128, 4, 6, 91, 91]);
    println!(
        "MLX-owned repeated I128 prefix, unread high-bit endpoint suffix and terminal tails: {result:?}"
    );
}

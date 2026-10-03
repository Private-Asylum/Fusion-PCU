//! Ordinary source repeats one readonly input while leaving another declaration unread.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuOwnedDispatchBackend,
    PcuF16Bits,
};
#[pcu(crate_path=::pcu_facade,invocations=N)]
fn square<T: PcuCheckedFloat, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}
fn main() {
    let (_, backend, _) = selection::selected_device();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(backend.device_identity().device_id()),
        ..Default::default()
    })
    .unwrap();
    let input = [PcuF16Bits::from_bits(0xbc00); 65];
    let mut output = [PcuF16Bits::from_bits(0x3555); 67];
    square::<_, 65>(&[], &mut output, &input).unwrap();
    assert!(output[..65].iter().all(|value| value.to_bits() == 0x3c00));
    assert!(output[65..].iter().all(|value| value.to_bits() == 0x3555));
    println!("65 exact F16 squares with an empty unread declaration; both tail elements preserved");
}

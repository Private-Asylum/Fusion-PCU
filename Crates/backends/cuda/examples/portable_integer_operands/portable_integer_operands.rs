//! Ordinary source repeats one readonly input while leaving another declaration unread.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedInteger,
    PcuOwnedDispatchBackend,
    PcuI512,
};
#[pcu(crate_path=::pcu_facade,invocations=N)]
fn square<T: PcuCheckedInteger, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}
fn main() {
    let (_, backend, _) = selection::selected_device();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(backend.device_identity().device_id()),
        ..Default::default()
    })
    .unwrap();
    let input = [PcuI512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]); 65];
    let mut output = [PcuI512::from_limbs_le([17, 0, 0, 0, 0, 0, 0, 0]); 67];
    square::<_, 65>(&[], &mut output, &input).unwrap();
    assert!(
        output[..65]
            .iter()
            .all(|value| *value == PcuI512::from_limbs_le([4, 0, 0, 0, 0, 0, 0, 0]))
    );
    assert!(
        output[65..]
            .iter()
            .all(|value| *value == PcuI512::from_limbs_le([17, 0, 0, 0, 0, 0, 0, 0]))
    );
    println!(
        "65 exact I512 squares with an empty unread declaration; both tail elements preserved"
    );
}

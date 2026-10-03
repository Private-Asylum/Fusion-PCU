//! Genuine source identity preserves every high limb and output tail.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/wide_transport/source/source.rs"]
#[allow(dead_code)] // Other matched boundaries are canonical benchmark fixtures.
mod source;
#[rustfmt::skip]
use fusion_pcu::{global,PcuScalar,PcuU512,PcuOwnedDispatchBackend};
fn main() {
    let (_, backend, _) = selection::selected_device();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(backend.device_identity().device_id()),
        ..Default::default()
    })
    .unwrap();
    let input = std::array::from_fn::<_, 65, _>(|index| {
        PcuU512::from_limbs_le(std::array::from_fn(|limb| {
            u64::try_from(index)
                .unwrap()
                .wrapping_mul(0x9e37)
                .wrapping_add(u64::try_from(limb).unwrap())
        }))
    });
    let sentinel = PcuU512::from_limbs_le([u64::MAX; 8]);
    let mut output = [sentinel; 67];
    source::grid::<PcuU512, 65>(&input, &mut output).unwrap();
    for (actual, want) in output.iter().zip(input) {
        assert_eq!(actual.encode_le(), want.encode_le());
    }
    assert_eq!(output[65..], [sentinel; 2]);
    println!("512-bit generic source identity preserved all eight limbs and output tails");
}

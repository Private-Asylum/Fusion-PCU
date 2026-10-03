//! Explicit integer Clamp publishes the saturated512-bit endpoint and reports recovery.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/integer_clamp/source/source.rs"]
#[allow(dead_code)] // Canonical fixture shares all ordinary source paths.
mod source;
#[rustfmt::skip]
use fusion_pcu::{global,PcuI512,PcuExecutionError,PcuExecutionFaultKind,PcuOwnedDispatchBackend};
fn main() {
    let (_, backend, _) = selection::selected_device();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(backend.device_identity().device_id()),
        ..Default::default()
    })
    .unwrap();
    let minus_one = PcuI512::from_limbs_le([u64::MAX; 8]);
    let mut out = [PcuI512::ZERO; 67];
    let result = source::mul::grid::<PcuI512, 65>(&[PcuI512::MIN; 65], &[minus_one; 65], &mut out);
    assert!(
        matches!(result,Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==0&&f.kind==PcuExecutionFaultKind::ArithmeticOverflow&&f.recovered)
    );
    assert_eq!(out[..65], [PcuI512::MAX; 65]);
    assert_eq!(out[65..], [PcuI512::ZERO; 2]);
    println!(
        "512-bit Clamp published maximum endpoint with observable recovered overflow; complete tails preserved"
    );
}

//! Checked signed512-bit multiply: exact boundary then fatal overflow rolls back output.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/wide_integer/source/source.rs"]
#[allow(dead_code)] // The canonical benchmark uses the other generic source paths.
mod source;
#[rustfmt::skip]
use fusion_pcu::{global,PcuI512,PcuExecutionError,PcuExecutionFaultKind,PcuOwnedDispatchBackend};
fn main() {
    let (_, backend, _) = selection::selected_device();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(backend.device_identity().device_id()),
        ..Default::default()
    })
    .unwrap();
    let input = [PcuI512::MIN; 65];
    let one = PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]);
    let mut output = [PcuI512::ZERO; 67];
    source::mul::grid::<PcuI512, 65>(&input, &[one; 65], &mut output).unwrap();
    assert_eq!(output[..65], input);
    assert_eq!(output[65..], [PcuI512::ZERO; 2]);
    let previous = output;
    let minus_one = PcuI512::from_limbs_le([u64::MAX; 8]);
    let result = source::mul::grid::<PcuI512, 65>(&input, &[minus_one; 65], &mut output);
    assert!(
        matches!(result,Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==0&&f.kind==PcuExecutionFaultKind::ArithmeticOverflow&&!f.recovered)
    );
    assert_eq!(output, previous);
    println!(
        "512-bit minimum multiply exact; minimum times negative-one overflow preserved complete destination"
    );
}

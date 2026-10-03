//! Actual annotated locals and both ordered outputs, explicitly routed to cuda.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/ordered_float_maps/source/source.rs"]
#[allow(dead_code)] // This small example uses the direct entry; the benchmark qualifies grid.
mod source;
#[rustfmt::skip]
use fusion_pcu::global::{
    self,
    PcuBackendChoice,
    PcuExecutionPolicy,
};
use fusion_pcu::PcuOwnedDispatchBackend;
fn main() {
    let (_, backend, _) = selection::selected_device();
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Cuda,
        device: Some(backend.device_identity().device_id()),
        ..Default::default()
    })
    .unwrap();
    let input = [0.5_f64, -1.0, 2.0];
    let mut stage = [99.0_f64; 5];
    let mut output = stage;
    source::direct::<f64, 3>(&mut stage, &mut output, &input).unwrap();
    assert_eq!(
        stage.map(f64::to_bits),
        [1.0_f64, -2.0, 4.0, 99.0, 99.0].map(f64::to_bits)
    );
    assert_eq!(
        output.map(f64::to_bits),
        [0.5_f64, 2.0, 8.0, 99.0, 99.0].map(f64::to_bits)
    );
    println!("cuda genuine scalar locals and ordered two-output stores passed");
}

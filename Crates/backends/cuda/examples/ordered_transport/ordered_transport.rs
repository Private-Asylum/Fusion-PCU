//! Genuine byte-preserving ordered transport, explicitly selecting cuda.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/ordered_transport/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/ordered_transport/source/source.rs"]
#[allow(dead_code)] // Both entries retain generated cold IR helpers for qualification.
mod source;
#[rustfmt::skip]
use fusion_pcu::global::{
    self,
    PcuBackendChoice,
    PcuExecutionPolicy,
};
use fusion_pcu::PcuOwnedDispatchBackend;
use oracle::Format;
fn demonstrate<T: Format>() {
    let input = [T::pattern(3), T::pattern(17), T::pattern(127)];
    let mut ghost = [];
    let mut stage = [T::pattern(251); 5];
    let mut output = stage;
    source::direct::<T, 3>(&input, &mut ghost, &mut stage, &mut output).unwrap();
    oracle::verify(&input, &stage);
    oracle::verify(&input, &output);
    source::grid::<T, 3>(&input, &mut ghost, &mut stage, &mut output).unwrap();
    oracle::verify(&input, &stage);
    oracle::verify(&input, &output);
}
fn main() {
    let (_, backend, _) = selection::selected_device();
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Cuda,
        device: Some(backend.device_identity().device_id()),
        ..Default::default()
    })
    .unwrap();
    demonstrate::<f64>();
    demonstrate::<fusion_pcu::PcuU512>();
    println!("cuda genuine ordered raw-bit transport: F64 payloads and all512 integer bits passed");
}

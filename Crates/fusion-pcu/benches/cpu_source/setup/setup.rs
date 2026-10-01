//! Cold configuration, heap-backed const-sized inputs, and diagnostic adaptation.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuNumericalMode,
};
use super::native::Error;

pub fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_mode: PcuNumericalMode::Strict,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
pub fn array<T: Clone, const N: usize>(value: T) -> Box<[T; N]> {
    std::vec![value; N]
        .into_boxed_slice()
        .try_into()
        .ok()
        .unwrap()
}
pub const fn global_error(error: &global::PcuExecutionError) -> Error {
    match error {
        global::PcuExecutionError::ArithmeticFault(fault) => Error::Fault(*fault),
        _ => Error::Schema,
    }
}
pub fn prepared_error(error: fusion_pcu_cpu::PcuCpuHostError) -> Error {
    error.fault().map_or(Error::Schema, Error::Fault)
}

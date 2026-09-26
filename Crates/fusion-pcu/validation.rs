//! Backend-neutral validation helpers for PCU model payloads.

mod command;
mod graphics;
mod stream;

pub use command::{
    validate_command_kernel,
    PcuCommandValidationError,
};
pub use graphics::{
    validate_sample_op,
    validate_trace_ray_op,
    PcuSampleValidationError,
    PcuTraceRayValidationError,
};
pub use stream::{
    validate_stream_simple_transform,
    PcuStreamSimpleTransformValidationError,
};

use crate::{
    PcuBinding,
    PcuBindingRef,
};

fn find_binding<'a>(
    bindings: &'a [PcuBinding<'a>],
    reference: PcuBindingRef,
) -> Option<PcuBinding<'a>> {
    bindings
        .iter()
        .copied()
        .find(|binding| binding.reference() == reference)
}

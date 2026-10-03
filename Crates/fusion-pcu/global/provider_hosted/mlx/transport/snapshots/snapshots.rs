//! Only old contents needed by validated ordered loads become physical inputs.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxPreparedTransportHostKernel,
    MlxTransportCompletion,
    MlxTransportInput,
};
#[rustfmt::skip]
use crate::{
    global::arguments::PcuCallArgumentKind,
    PcuExecutionError,
};
use super::super::map_mlx_error;

pub(super) fn execute<const N: usize>(
    kernel: &mut MlxPreparedTransportHostKernel,
    arguments: &[PcuCallArgumentKind<'_>; N],
) -> Result<MlxTransportCompletion, PcuExecutionError> {
    let mut inputs = [None; 4];
    let count = kernel.plan().inputs().len();
    for (slot, resource) in kernel.plan().inputs().iter().enumerate() {
        let kind = arguments
            .iter()
            .find(|kind| super::preflight::target(kind).ok() == Some(resource.binding))
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        inputs[slot] = Some(match kind {
            PcuCallArgumentKind::Host(argument) => MlxTransportInput::HostBytes {
                target: resource.binding,
                scalar: argument.scalar(),
                bytes: argument.bytes(),
            },
            PcuCallArgumentKind::MlxRead(argument) => MlxTransportInput::Resident {
                target: resource.binding,
                array: argument.array,
            },
            PcuCallArgumentKind::MlxWrite(argument) => MlxTransportInput::Resident {
                target: resource.binding,
                array: argument.array,
            },
            #[cfg(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            ))]
            _ => return Err(PcuExecutionError::InvalidTensorSourcePlan),
        });
    }
    let first = inputs[0].ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
    let ordered = inputs.map(|input| input.unwrap_or(first));
    kernel
        .execute_inputs(&ordered[..count])
        .map_err(map_mlx_error)
}

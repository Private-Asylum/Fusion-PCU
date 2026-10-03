//! Fallible sibling materialization/merging/releases finish before any public write.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxEncodedArray,
    MlxPreparedEncodedPrefix,
    MlxPreparedTransportHostKernel,
    MlxTransportCompletion,
};
#[rustfmt::skip]
use crate::{
    global::arguments::PcuCallArgumentKind,
    PcuExecutionError,
    PcuBindingRef,
};
use super::super::map_mlx_execution;

pub(super) fn publish<const N: usize>(
    kernel: &MlxPreparedTransportHostKernel,
    prefixes: &[Option<MlxPreparedEncodedPrefix>; 2],
    readback: &mut [Vec<u8>; 2],
    arguments: &mut [PcuCallArgumentKind<'_>; N],
    completed: MlxTransportCompletion,
) -> Result<(), PcuExecutionError> {
    let layout = kernel.output_layout();
    let mut outputs = completed.into_outputs();
    let mut host = [false; 2];
    for (slot, output) in outputs.iter_mut().enumerate() {
        let Some((binding, bytes)) = layout[slot] else {
            if output.is_some() {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            }
            continue;
        };
        let array = output
            .as_ref()
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        if array.scalar_type() != kernel.plan().scalar_type()
            || array.byte_len() != bytes
            || !array.same_session(kernel.session())
        {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let destination = arguments
            .iter()
            .find(|kind| super::preflight::target(kind).ok() == Some(binding))
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        match destination {
            PcuCallArgumentKind::Host(_) => {
                array
                    .read_bytes_into(&mut readback[slot])
                    .map_err(map_mlx_execution)?;
                host[slot] = true;
            }
            PcuCallArgumentKind::MlxWrite(destination) => {
                if let Some(prefix) = &prefixes[slot] {
                    let merged = prefix
                        .execute(array, destination.array)
                        .map_err(map_mlx_execution)?;
                    output
                        .take()
                        .expect("the validated private array exists")
                        .release()
                        .map_err(map_mlx_execution)?;
                    *output = Some(merged);
                }
            }
            PcuCallArgumentKind::MlxRead(_) => {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            }
            #[cfg(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            ))]
            _ => return Err(PcuExecutionError::InvalidTensorSourcePlan),
        }
    }
    for (slot, output) in outputs.iter_mut().enumerate() {
        if host[slot] {
            output
                .take()
                .expect("a validated host result exists")
                .release()
                .map_err(map_mlx_execution)?;
        }
    }
    commit(layout, readback, arguments, outputs);
    Ok(())
}

/// This boundary cannot return an error. Every native read/merge/release is
/// finished and complete preflight fixed the destinations and their capacities.
fn commit<const N: usize>(
    layout: [Option<(PcuBindingRef, usize)>; 2],
    readback: &[Vec<u8>; 2],
    arguments: &mut [PcuCallArgumentKind<'_>; N],
    mut outputs: [Option<MlxEncodedArray>; 2],
) {
    for argument in arguments {
        let binding =
            super::preflight::target(argument).expect("complete preflight accepted this kind");
        let Some(slot) = layout
            .iter()
            .position(|output| output.is_some_and(|(target, _)| target == binding))
        else {
            continue;
        };
        let (_, bytes) = layout[slot].expect("an actual output slot was found");
        match argument {
            PcuCallArgumentKind::Host(destination) => {
                destination
                    .bytes_mut()
                    .expect("preflight proved exclusive host access")[..bytes]
                    .copy_from_slice(&readback[slot]);
            }
            PcuCallArgumentKind::MlxWrite(destination) => {
                *destination.array = outputs[slot]
                    .take()
                    .expect("a terminal private owner exists");
            }
            PcuCallArgumentKind::MlxRead(_) => {
                unreachable!("preflight proved the writer's exclusive access")
            }
            #[cfg(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            ))]
            _ => unreachable!("preflight excluded foreign providers"),
        }
    }
}

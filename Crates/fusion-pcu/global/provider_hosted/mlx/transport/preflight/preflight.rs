//! Complete declaration validation precedes staging, kernel work and owner changes.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxPreparedEncodedPrefix,
    MlxPreparedTransportHostKernel,
};
#[rustfmt::skip]
use crate::{
    global::arguments::PcuCallArgumentKind,
    PcuBindingAccess,
    PcuBindingRef,
    PcuExecutionError,
    PcuHostDispatchError,
};
#[rustfmt::skip]
use super::super::{
    map_mlx_error,
    map_mlx_execution,
    validate_host,
    validate_output,
};

#[cfg_attr(
    not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    )),
    allow(clippy::unnecessary_wraps)
)] // Foreign-provider borrows are fallible when those variants are compiled in.
pub(super) const fn target(
    kind: &PcuCallArgumentKind<'_>,
) -> Result<PcuBindingRef, PcuExecutionError> {
    match kind {
        PcuCallArgumentKind::Host(argument) => Ok(argument.target()),
        PcuCallArgumentKind::MlxRead(argument) => Ok(argument.target),
        PcuCallArgumentKind::MlxWrite(argument) => Ok(argument.target),
        #[cfg(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            all(feature = "cpu", feature = "tensor"),
            all(feature = "vulkan", feature = "tensor")
        ))]
        _ => Err(PcuExecutionError::Argument(
            crate::global::PcuArgumentError::SessionMismatch,
        )),
    }
}

pub(super) fn validate<const N: usize>(
    kernel: &MlxPreparedTransportHostKernel,
    prefixes: &[Option<MlxPreparedEncodedPrefix>; 2],
    readback: &[Vec<u8>; 2],
    arguments: &[PcuCallArgumentKind<'_>; N],
) -> Result<(), PcuExecutionError> {
    kernel
        .session()
        .validate_access_available()
        .map_err(map_mlx_execution)?;
    let plan = kernel.plan();
    let scalar = plan.scalar_type();
    let width = usize::from(scalar.bit_width()) / 8;
    let mut seen = [PcuBindingRef::new(0, 0); N];
    for (position, kind) in arguments.iter().enumerate() {
        let binding = target(kind)?;
        if seen[..position].contains(&binding) {
            return Err(map_mlx_error(PcuHostDispatchError::Duplicate(binding)));
        }
        seen[position] = binding;
        let access = plan
            .declared_bindings()
            .iter()
            .find_map(|(target, access)| (*target == binding).then_some(*access))
            .ok_or_else(|| map_mlx_error(PcuHostDispatchError::Unexpected(binding)))?;
        let count = plan
            .resources()
            .iter()
            .find(|resource| resource.binding == binding)
            .map_or(0, |resource| resource.minimum_elements());
        let bytes =
            usize::try_from(count).map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)? * width;
        match kind {
            PcuCallArgumentKind::Host(argument) => validate_host(argument, scalar, access, bytes)?,
            PcuCallArgumentKind::MlxRead(argument) => {
                if access != PcuBindingAccess::ReadOnly {
                    return Err(map_mlx_error(PcuHostDispatchError::AccessMismatch(binding)));
                }
                validate_array(kernel, argument.array, binding, bytes)?;
            }
            PcuCallArgumentKind::MlxWrite(argument) => {
                if access != PcuBindingAccess::ReadWrite {
                    return Err(map_mlx_error(PcuHostDispatchError::AccessMismatch(binding)));
                }
                validate_array(kernel, argument.array, binding, bytes)?;
            }
            #[cfg(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            ))]
            _ => {
                return Err(PcuExecutionError::Argument(
                    crate::global::PcuArgumentError::SessionMismatch,
                ));
            }
        }
        if let Some(slot) = kernel
            .output_layout()
            .iter()
            .position(|output| output.is_some_and(|(target, _)| target == binding))
        {
            let (_, bytes) = kernel.output_layout()[slot].expect("an actual writer slot was found");
            validate_output(kind, scalar, bytes, binding, prefixes[slot].as_ref())?;
            if matches!(kind, PcuCallArgumentKind::Host(_)) && readback[slot].len() != bytes {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            }
        }
    }
    for (binding, _) in plan.declared_bindings() {
        if !seen.contains(binding) {
            return Err(map_mlx_error(PcuHostDispatchError::Missing(*binding)));
        }
    }
    Ok(())
}

fn validate_array(
    kernel: &MlxPreparedTransportHostKernel,
    array: &fusion_pcu_mlx::MlxEncodedArray,
    binding: PcuBindingRef,
    bytes: usize,
) -> Result<(), PcuExecutionError> {
    if array.scalar_type() != kernel.plan().scalar_type() {
        return Err(map_mlx_error(PcuHostDispatchError::TypeMismatch(binding)));
    }
    if array.byte_len() < bytes {
        return Err(map_mlx_error(PcuHostDispatchError::BufferTooSmall(binding)));
    }
    if !array.same_session(kernel.session()) {
        return Err(PcuExecutionError::Argument(
            crate::global::PcuArgumentError::SessionMismatch,
        ));
    }
    array.validate_access_available().map_err(map_mlx_execution)
}

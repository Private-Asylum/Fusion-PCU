//! Frozen binary input roles and terminal publication for ordinary Rust borrows.

#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedBinaryPlan,
    MlxBinaryInput,
    MlxPreparedEncodedPrefix,
    MlxSession,
};
#[rustfmt::skip]
use crate::{
    global::arguments::{
        PcuCallArgument,
        PcuCallArgumentKind,
    },
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuExecutionError,
    PcuHostDispatchError,
    PcuHostKernelBackend,
};
#[rustfmt::skip]
use super::{
    map_mlx_error,
    map_mlx_execution,
    publish_completed,
    readonly_declarations,
    validate_host,
    validate_output,
    two_input::TwoInputKernel,
    MlxInputLayout,
    MlxKernel,
};

pub(super) fn prepare(
    session: &MlxSession,
    source: &PcuDispatchKernelIr<'_>,
    inputs: MlxInputLayout,
    plan: &MlxCheckedBinaryPlan,
) -> Result<MlxKernel, PcuExecutionError> {
    let minimum = plan.input_element_counts();
    let extents = inputs.extents(plan.input_bindings(), minimum)?;
    let backend = session.checked_binary_backend();
    // Retain each unique native shape; broadcast/repeated logical spans stay
    // in the plan. Exact-minimum calls preserve their established constructor.
    let kernel = if extents == minimum {
        backend.prepare_host_kernel(source)
    } else {
        backend
            .prepare_host_kernel_with_input_extents(source, &extents[..plan.input_bindings().len()])
    }
    .map_err(map_mlx_error)?;
    let (declarations, declaration_count) =
        readonly_declarations(source, kernel.input_bindings()[0]);
    Ok(MlxKernel::Binary {
        kernel,
        declarations,
        declaration_count,
    })
}

pub(super) fn call<K: TwoInputKernel, const N: usize>(
    prepared: &mut K,
    declarations: &[PcuBindingRef],
    prefix: Option<&MlxPreparedEncodedPrefix>,
    arguments: [PcuCallArgument<'_>; N],
) -> Result<(), PcuExecutionError> {
    let (inputs, output) = collect(
        prepared.input_bindings(),
        declarations,
        prepared.output_binding(),
        prepared.scalar_type(),
        arguments,
    )?;
    let (inputs, mut output) = match (inputs, output) {
        (
            [
                Some(PcuCallArgumentKind::Host(left)),
                Some(PcuCallArgumentKind::Host(right)),
            ],
            PcuCallArgumentKind::Host(output),
        ) => {
            return prepared
                .call(&mut [left, right, output])
                .map_err(map_mlx_error);
        }
        ([Some(PcuCallArgumentKind::Host(input)), None], PcuCallArgumentKind::Host(output)) => {
            return prepared.call(&mut [input, output]).map_err(map_mlx_error);
        }
        pair => pair,
    };
    let scalar = prepared.scalar_type();
    let bytes = prepared.output_element_count() * (usize::from(scalar.bit_width()) / 8);
    validate_output(&output, scalar, bytes, prepared.output_binding(), prefix)?;
    let mut borrowed = [None; 2];
    for (slot, kind) in inputs
        .iter()
        .enumerate()
        .take(prepared.input_bindings().len())
    {
        borrowed[slot] = Some(input(
            prepared,
            slot,
            kind.as_ref()
                .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
        )?);
    }
    let first = borrowed[0].ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
    let completion = if let Some(second) = borrowed[1] {
        prepared.execute_inputs(&[first, second])
    } else {
        prepared.execute_inputs(&[first])
    }
    .map_err(map_mlx_execution)?;
    let (completed, recovered) = completion.into_parts();
    publish_completed(&mut output, completed, scalar, bytes, prefix)?;
    recovered.map_or(Ok(()), |fault| {
        Err(PcuExecutionError::ArithmeticFault(fault))
    })
}

fn input<'a, K: TwoInputKernel>(
    prepared: &K,
    slot: usize,
    kind: &'a PcuCallArgumentKind<'_>,
) -> Result<MlxBinaryInput<'a>, PcuExecutionError> {
    let target = prepared.input_bindings()[slot];
    match kind {
        PcuCallArgumentKind::Host(argument) => {
            let scalar = prepared.scalar_type();
            let bytes =
                prepared.input_element_counts()[slot] * (usize::from(scalar.bit_width()) / 8);
            validate_host(argument, scalar, PcuBindingAccess::ReadOnly, bytes)?;
            Ok(MlxBinaryInput::HostBytes {
                target,
                scalar,
                bytes: argument.bytes(),
            })
        }
        PcuCallArgumentKind::MlxRead(argument) => Ok(MlxBinaryInput::Resident {
            target,
            array: argument.array,
        }),
        PcuCallArgumentKind::MlxWrite(_) => {
            Err(map_mlx_error(PcuHostDispatchError::AccessMismatch(target)))
        }
        #[cfg(all(feature = "vulkan", feature = "tensor"))]
        PcuCallArgumentKind::VulkanRead(_) | PcuCallArgumentKind::VulkanWrite(_) => Err(
            PcuExecutionError::Argument(crate::global::PcuArgumentError::SessionMismatch),
        ),
        #[cfg(all(feature = "cpu", feature = "tensor"))]
        PcuCallArgumentKind::CpuOwner(_) => Err(PcuExecutionError::Argument(
            crate::global::PcuArgumentError::SessionMismatch,
        )),
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        PcuCallArgumentKind::ResidentRead(_) | PcuCallArgumentKind::ResidentWrite(_) => Err(
            PcuExecutionError::Argument(crate::global::PcuArgumentError::SessionMismatch),
        ),
    }
}

type Inputs<'a> = [Option<PcuCallArgumentKind<'a>>; 2];

fn collect<'a, const N: usize>(
    input_bindings: &[PcuBindingRef],
    declarations: &[PcuBindingRef],
    output_binding: PcuBindingRef,
    scalar: crate::PcuScalarType,
    arguments: [PcuCallArgument<'a>; N],
) -> Result<(Inputs<'a>, PcuCallArgumentKind<'a>), PcuExecutionError> {
    if N != declarations.len() + 1 || N > 3 {
        return Err(PcuExecutionError::InvalidTensorSourcePlan);
    }
    let mut inputs = [None, None];
    let mut output = None;
    let mut seen = [output_binding; 3];
    for (position, argument) in arguments.into_iter().enumerate() {
        let kind = argument.into_parts().1;
        let target = match &kind {
            PcuCallArgumentKind::Host(argument) => argument.target(),
            PcuCallArgumentKind::MlxRead(argument) => argument.target,
            PcuCallArgumentKind::MlxWrite(argument) => argument.target,
            #[cfg(all(feature = "vulkan", feature = "tensor"))]
            PcuCallArgumentKind::VulkanRead(_) | PcuCallArgumentKind::VulkanWrite(_) => {
                return Err(PcuExecutionError::Argument(
                    crate::global::PcuArgumentError::SessionMismatch,
                ));
            }
            #[cfg(all(feature = "cpu", feature = "tensor"))]
            PcuCallArgumentKind::CpuOwner(_) => {
                return Err(PcuExecutionError::Argument(
                    crate::global::PcuArgumentError::SessionMismatch,
                ));
            }
            #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
            _ => {
                return Err(PcuExecutionError::Argument(
                    crate::global::PcuArgumentError::SessionMismatch,
                ));
            }
        };
        if seen[..position].contains(&target) {
            return Err(map_mlx_error(PcuHostDispatchError::Duplicate(target)));
        }
        seen[position] = target;
        if target == output_binding {
            output = Some(kind);
        } else if let Some(slot) = input_bindings.iter().position(|input| *input == target) {
            inputs[slot] = Some(kind);
        } else if declarations.contains(&target) {
            // An unused source parameter must keep its declared readonly/type role,
            // but requires no upload, element access or fabricated second allocation.
            match &kind {
                PcuCallArgumentKind::Host(argument) => {
                    validate_host(argument, scalar, PcuBindingAccess::ReadOnly, 0)?;
                }
                PcuCallArgumentKind::MlxRead(argument)
                    if argument.array.scalar_type() == scalar => {}
                _ => return Err(map_mlx_error(PcuHostDispatchError::AccessMismatch(target))),
            }
        } else {
            return Err(map_mlx_error(PcuHostDispatchError::Unexpected(target)));
        }
    }
    for (slot, target) in input_bindings.iter().enumerate() {
        if inputs[slot].is_none() {
            return Err(map_mlx_error(PcuHostDispatchError::Missing(*target)));
        }
    }
    let output =
        output.ok_or_else(|| map_mlx_error(PcuHostDispatchError::Missing(output_binding)))?;
    Ok((inputs, output))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

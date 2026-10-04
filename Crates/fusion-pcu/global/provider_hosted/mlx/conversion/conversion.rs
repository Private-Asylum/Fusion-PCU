//! Mixed-width ordinary conversion without homogeneous input/output assumptions.

#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxBinaryInput,
    MlxCheckedConversionPlan,
    MlxConversionHostBackend,
    MlxPreparedConversionHostKernel,
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
    PcuDispatchKernelIr,
    PcuExecutionError,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
#[rustfmt::skip]
use super::{
    map_mlx_error,
    map_mlx_execution,
    publish_completed,
    single_input,
    validate_host,
    validate_output,
    MlxInputLayout,
    MlxKernel,
};

pub(super) fn prepare(
    session: &MlxSession,
    source: &PcuDispatchKernelIr<'_>,
    inputs: MlxInputLayout,
    plan: MlxCheckedConversionPlan,
) -> Result<MlxKernel, PcuExecutionError> {
    let minimum = [plan.input_element_count(), 0];
    let extents = inputs.extents(&[plan.input_binding()], minimum)?;
    let backend = MlxConversionHostBackend::new(session.clone());
    // Actual resident shape is part of cold cache identity. Replay neither
    // reshapes the input nor stages it through RAM; only logical lanes are read.
    let kernel = if extents == minimum {
        backend.prepare_host_kernel(source)
    } else {
        backend.prepare_host_kernel_with_input_extent(source, extents[0])
    }
    .map_err(map_mlx_execution)?;
    Ok(MlxKernel::Conversion(kernel))
}

pub(super) fn call<const N: usize>(
    kernel: &mut MlxPreparedConversionHostKernel,
    prefix: Option<&MlxPreparedEncodedPrefix>,
    arguments: [PcuCallArgument<'_>; N],
) -> Result<(), PcuExecutionError> {
    let plan = kernel.plan();
    let (input, output) = single_input::collect(
        plan.input_binding(),
        plan.output_binding(),
        plan.source_scalar(),
        &[],
        arguments,
    )?;
    let (input, mut output) = match (input, output) {
        (PcuCallArgumentKind::Host(input), PcuCallArgumentKind::Host(output)) => {
            return kernel.call(&mut [input, output]).map_err(map_mlx_execution);
        }
        pair => pair,
    };
    validate_output(
        &output,
        plan.output_scalar(),
        kernel.output_byte_len(),
        plan.output_binding(),
        prefix,
    )?;
    let operand = input_operand(&input, kernel)?;
    let completion = kernel
        .execute_inputs(&[operand])
        .map_err(map_mlx_execution)?;
    if let PcuCallArgumentKind::Host(destination) = &mut output {
        let bytes = destination
            .bytes_mut()
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        return kernel
            .publish_host_completion(completion, bytes)
            .map_err(map_mlx_execution);
    }
    let (completed, notice) = completion.into_parts();
    publish_completed(
        &mut output,
        completed,
        plan.output_scalar(),
        kernel.output_byte_len(),
        prefix,
    )?;
    notice.map_or(Ok(()), |fault| {
        Err(PcuExecutionError::ArithmeticFault(fault))
    })
}

fn input_operand<'a>(
    input: &'a PcuCallArgumentKind<'_>,
    kernel: &MlxPreparedConversionHostKernel,
) -> Result<MlxBinaryInput<'a>, PcuExecutionError> {
    let target = kernel.plan().input_binding();
    match input {
        PcuCallArgumentKind::Host(argument) => {
            validate_host(
                argument,
                kernel.source_scalar_type(),
                PcuBindingAccess::ReadOnly,
                kernel.input_byte_len(),
            )?;
            Ok(MlxBinaryInput::HostBytes {
                target,
                scalar: kernel.source_scalar_type(),
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

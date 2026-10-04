//! Cold MLX inventory and static host execution; no implicit different-provider delegation.
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCarrierPlan,
    MlxCheckedBinaryPlan,
    MlxCheckedUnaryPlan,
    MlxCheckedIntegerPlan,
    MlxCheckedDivRemPlan,
    MlxCheckedDivRemRolePlan,
    MlxDiscovery,
    MlxError,
    MlxHostKernelError,
    MlxPreparedEncodedPrefix,
    MlxPreparedHostKernel,
    MlxPreparedBinaryHostKernel,
    MlxPreparedIntegerHostKernel,
    MlxPreparedDivRemHostKernel,
    MlxPreparedDivRemRoleHostKernel,
    MlxPreparedCarrierHostKernel,
    MlxPreparedTransportHostKernel,
    MlxPreparedConversionHostKernel,
    MlxCheckedConversionPlan,
    MlxSession,
};
#[rustfmt::skip]
use super::{
    Candidate,
    EMPTY_READY,
    EMPTY_REF,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuExecutionError,
    PcuExecutionPolicy,
    PcuObjectRef,
    PcuPreparedHostKernel,
    PcuHostKernelBackend,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuRuntimeDiscovery,
    PcuTargetDescriptor,
    Provider,
    Session,
};

#[path = "binary/binary.rs"]
mod binary;
#[path = "integer/integer.rs"]
mod integer;
#[path = "output/output.rs"]
mod output;
pub(super) use output::MlxOutputLayout;
#[path = "input/input.rs"]
mod input;
pub(super) use input::MlxInputLayout;
#[path = "single_input/single_input.rs"]
mod single_input;
use single_input::SingleInputKernel;

#[path = "conversion/conversion.rs"]
mod conversion;
#[path = "div_rem/div_rem.rs"]
mod div_rem;
#[path = "transport/transport.rs"]
mod transport;
#[path = "two_input/two_input.rs"]
mod two_input;

// Concrete paths stay inline in the retained cache; warm selection is an enum match.
#[allow(clippy::large_enum_variant)] // Boxing the cold prepared kernel adds unnecessary warm ownership indirection.
pub(super) enum MlxKernel {
    Conversion(MlxPreparedConversionHostKernel),
    Carrier(MlxPreparedCarrierHostKernel),
    Transport {
        kernel: MlxPreparedTransportHostKernel,
        // Mixed calls materialize host siblings privately. Ordinary all-host
        // calls keep the backend's original retained readback path.
        readback: [Vec<u8>; 2],
    },
    Unary(MlxPreparedHostKernel),
    Binary {
        kernel: MlxPreparedBinaryHostKernel,
        declarations: [crate::PcuBindingRef; 2],
        declaration_count: usize,
    },
    Integer {
        kernel: MlxPreparedIntegerHostKernel,
        declarations: [crate::PcuBindingRef; 2],
        declaration_count: usize,
    },
    DivRemRoles {
        kernel: MlxPreparedDivRemRoleHostKernel,
        declarations: [crate::PcuBindingRef; 2],
        declaration_count: usize,
        readback: [Vec<u8>; 2],
    },
    DivRem {
        kernel: MlxPreparedDivRemHostKernel,
        // Host materialization is private until both siblings complete. These
        // exact-sized buffers are allocated once during cold preparation.
        readback: [Vec<u8>; 2],
    },
}
impl MlxKernel {
    pub(super) fn call(
        &mut self,
        args: &mut [crate::PcuHostArgument<'_>],
    ) -> Result<(), MlxHostKernelError> {
        match self {
            Self::Conversion(kernel) => kernel
                .call(args)
                .map_err(crate::PcuHostDispatchError::Backend),
            Self::Carrier(kernel) => kernel.call(args),
            Self::Transport { kernel, .. } => kernel.call(args),
            Self::Unary(kernel) => kernel.call(args),
            Self::Binary { kernel, .. } => kernel.call(args),
            Self::Integer { kernel, .. } => kernel.call(args),
            Self::DivRem { kernel, .. } => kernel.call(args),
            Self::DivRemRoles { kernel, .. } => kernel.call(args),
        }
    }
    const fn scalar_type(&self) -> crate::PcuScalarType {
        match self {
            Self::Conversion(kernel) => kernel.scalar_type(),
            Self::Carrier(kernel) => kernel.scalar_type(),
            Self::Transport { kernel, .. } => kernel.plan().scalar_type(),
            Self::Unary(kernel) => kernel.scalar_type(),
            Self::Binary { kernel, .. } => kernel.scalar_type(),
            Self::Integer { kernel, .. } => kernel.scalar_type(),
            Self::DivRem { kernel, .. } => kernel.scalar_type(),
            Self::DivRemRoles { kernel, .. } => kernel.scalar_type(),
        }
    }
    fn outputs(&self) -> [Option<(crate::PcuBindingRef, usize)>; 2] {
        let single = |binding, bytes| [Some((binding, bytes)), None];
        match self {
            Self::Conversion(kernel) => single(kernel.output_binding(), kernel.output_byte_len()),
            Self::Carrier(kernel) => single(kernel.output_binding(), kernel.output_byte_len()),
            Self::Transport { kernel, .. } => kernel.output_layout(),
            Self::Unary(kernel) => single(kernel.output_binding(), kernel.output_byte_len()),
            Self::Binary { kernel, .. } => single(
                kernel.output_binding(),
                kernel.output_element_count() * (usize::from(kernel.scalar_type().bit_width()) / 8),
            ),
            Self::Integer { kernel, .. } => {
                single(kernel.output_binding(), kernel.output_byte_len())
            }
            Self::DivRem { kernel, .. } => {
                let bindings = kernel.output_bindings();
                let bytes = kernel.output_byte_lengths();
                [Some((bindings[0], bytes[0])), Some((bindings[1], bytes[1]))]
            }
            Self::DivRemRoles { kernel, .. } => {
                let bindings = kernel.output_bindings();
                let bytes = kernel.output_byte_lengths();
                [Some((bindings[0], bytes[0])), Some((bindings[1], bytes[1]))]
            }
        }
    }
}

/// Actual resident extent participates in cold specialization. A new output shape
/// prepares its merge before execution; publication never compiles a native operation.
pub(in crate::global) struct MlxInvocation {
    pub(super) kernel: MlxKernel,
    prefixes: [Option<MlxPreparedEncodedPrefix>; 2],
}
impl MlxInvocation {
    pub(super) fn prepare(
        session: &MlxSession,
        source: &crate::PcuDispatchKernelIr<'_>,
        inputs: MlxInputLayout,
    ) -> Result<Self, PcuExecutionError> {
        let kernel = if let Ok(plan) = MlxCheckedConversionPlan::assess(source) {
            conversion::prepare(session, source, inputs, plan)?
        } else if let Ok(plan) = MlxCarrierPlan::assess(source) {
            let minimum = [plan.input_element_count(), 0];
            let extents = inputs.extents(plan.input_bindings(), minimum)?;
            // Logical read spans remain unchanged. An escaped MLX input keeps
            // its full physical shape, frozen here rather than reshaped warm.
            let prepared = if extents == minimum {
                session.prepare_carrier_host_kernel(source)
            } else {
                session.prepare_carrier_host_kernel_with_input_extents(
                    source,
                    &extents[..plan.input_bindings().len()],
                )
            }
            .map_err(map_mlx_error)?;
            MlxKernel::Carrier(prepared)
        } else if let Ok(plan) = MlxCheckedBinaryPlan::assess(source) {
            binary::prepare(session, source, inputs, &plan)?
        } else if let Ok(plan) = MlxCheckedIntegerPlan::assess(source) {
            integer::prepare(session, source, inputs, &plan)?
        } else if let Ok(plan) = MlxCheckedDivRemPlan::assess(source) {
            let minimum = [plan.element_count(); 2];
            let extents = inputs.extents(plan.input_bindings(), minimum)?;
            let backend = session.checked_div_rem_backend();
            let kernel = if extents == minimum {
                backend.prepare_host_kernel(source)
            } else {
                backend.prepare_host_kernel_with_input_extents(source, &extents)
            }
            .map_err(map_mlx_error)?;
            let readback = [Vec::new(), Vec::new()];
            MlxKernel::DivRem { kernel, readback }
        } else if let Ok(plan) = MlxCheckedDivRemRolePlan::assess(source) {
            let minimum = plan.input_element_counts();
            let extents = inputs.extents(plan.input_bindings(), minimum)?;
            let backend = session.checked_div_rem_role_backend();
            let kernel = if extents == minimum {
                backend.prepare_host_kernel(source)
            } else {
                backend.prepare_host_kernel_with_input_extents(
                    source,
                    &extents[..plan.input_bindings().len()],
                )
            }
            .map_err(map_mlx_error)?;
            let (declarations, declaration_count) =
                readonly_declarations(source, kernel.input_bindings()[0]);
            MlxKernel::DivRemRoles {
                kernel,
                declarations,
                declaration_count,
                readback: [Vec::new(), Vec::new()],
            }
        } else if let Some(crate::PcuBindingType::Value(crate::PcuValueType::Scalar(scalar))) =
            source.bindings.first().map(|binding| binding.binding_type)
            && let Ok(description) = crate::describe_scalar_transport_map::<4>(source, scalar)
        {
            let mut bindings = [crate::PcuBindingRef::new(0, 0); 4];
            let mut minimum = [0; 4];
            let mut count = 0;
            for resource in description.resources() {
                if resource.minimum_initial_read_elements != 0 {
                    bindings[count] = resource.binding;
                    minimum[count] = usize::try_from(resource.minimum_initial_read_elements)
                        .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)?;
                    count += 1;
                }
            }
            let extents = inputs.snapshot_extents(&bindings[..count], minimum)?;
            let backend = session.transport_host_backend();
            let kernel = if extents == minimum {
                backend.prepare_host_kernel(source)
            } else {
                backend.prepare_host_kernel_with_input_extents(source, &extents[..count])
            }
            .map_err(map_mlx_error)?;
            MlxKernel::Transport {
                kernel,
                readback: [Vec::new(), Vec::new()],
            }
        } else {
            let plan = MlxCheckedUnaryPlan::assess(source)
                .map_err(|error| map_mlx_error(crate::PcuHostDispatchError::Backend(error)))?;
            let minimum = [plan.input_element_count(), 0];
            let extents = inputs.extents(plan.input_bindings(), minimum)?;
            // Keep the logical load span and freeze an escaped owner's full
            // native shape cold. Exact-minimum host calls keep their old path.
            let prepared = if extents == minimum {
                session.prepare_unary_host_kernel(source)
            } else {
                session.prepare_unary_host_kernel_with_input_extents(
                    source,
                    &extents[..plan.input_bindings().len()],
                )
            }
            .map_err(map_mlx_error)?;
            MlxKernel::Unary(prepared)
        };
        Ok(Self {
            kernel,
            prefixes: [None, None],
        })
    }

    pub(super) fn prepare_outputs(
        &mut self,
        session: &MlxSession,
        layout: MlxOutputLayout,
        resident: bool,
    ) -> Result<(), PcuExecutionError> {
        let outputs = self.kernel.outputs();
        // Ordered transport can read an exclusive bank without writing it.
        // Its complete declaration/resource preflight validates those borrows;
        // only actual writer bindings participate in publication.
        if !matches!(self.kernel, MlxKernel::Transport { .. }) {
            layout.validate(outputs)?;
        }
        let width = usize::from(self.kernel.scalar_type().bit_width()) / 8;
        for (slot, output) in outputs.into_iter().enumerate() {
            let Some((binding, bytes)) = output else {
                continue;
            };
            let Some(output_count) = layout.count(binding) else {
                continue;
            };
            let prefix_count = bytes / width;
            if output_count < prefix_count {
                return Err(map_mlx_error(crate::PcuHostDispatchError::BufferTooSmall(
                    binding,
                )));
            }
            if output_count > prefix_count {
                self.prefixes[slot] = Some(
                    session
                        .prepare_encoded_prefix(
                            self.kernel.scalar_type(),
                            prefix_count,
                            output_count,
                        )
                        .map_err(map_mlx_execution)?,
                );
            }
        }
        if let MlxKernel::DivRem { readback, .. } | MlxKernel::DivRemRoles { readback, .. } =
            &mut self.kernel
        {
            for (slot, output) in outputs.into_iter().enumerate() {
                if let Some((binding, bytes)) = output
                    && resident
                    && layout.count(binding).is_none()
                {
                    readback[slot] = vec![0; bytes];
                }
            }
        }
        if let MlxKernel::Transport { readback, .. } = &mut self.kernel {
            for (slot, output) in outputs.into_iter().enumerate() {
                if let Some((binding, bytes)) = output
                    && resident
                    && layout.count(binding).is_none()
                {
                    readback[slot] = vec![0; bytes];
                }
            }
        }
        Ok(())
    }
}

// Called only after exact binary/integer/joint assessment proves one or two readonly
// declarations. Declaration order remains source metadata, not operand order.
fn readonly_declarations(
    source: &crate::PcuDispatchKernelIr<'_>,
    first: crate::PcuBindingRef,
) -> ([crate::PcuBindingRef; 2], usize) {
    let mut declarations = [first; 2];
    let mut count = 0;
    for binding in source.bindings {
        if binding.access == crate::PcuBindingAccess::ReadOnly {
            declarations[count] = binding.reference();
            count += 1;
        }
    }
    (declarations, count)
}

pub(super) fn map_mlx_error(error: MlxHostKernelError) -> PcuExecutionError {
    match error {
        crate::PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)) => {
            PcuExecutionError::ArithmeticFault(fault)
        }
        other => PcuExecutionError::MlxHostExecution(other),
    }
}

pub(super) fn open_mlx(
    discovery: Option<&Result<MlxDiscovery, MlxError>>,
    device: PcuObjectRef,
) -> Result<Session, PcuExecutionError> {
    let discovery = discovery
        .and_then(|result| result.as_ref().ok())
        .ok_or(PcuExecutionError::NoBackendEnabled)?;
    crate::PcuDeviceActivation::open_device(discovery, device)
        .map(|session| Session::Mlx(Rc::new(session)))
        .map_err(PcuExecutionError::MlxExecution)
}

pub(super) fn collect_mlx_candidates(
    discovery: &MlxDiscovery,
    policy: PcuExecutionPolicy,
    kernel: &crate::PcuDispatchKernelIr<'_>,
    output: &mut Vec<Candidate>,
) -> Result<(), MlxError> {
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: EMPTY_READY,
    }];
    discovery.providers(&mut providers)?;
    let mut targets = [PcuTargetDescriptor {
        reference: EMPTY_REF,
        name: "",
        readiness: EMPTY_READY,
    }];
    discovery.targets(providers[0].id, providers[0].generation, &mut targets)?;
    let count = discovery.devices(targets[0].reference, &mut [])?;
    let blank = PcuDeviceDescriptor {
        reference: EMPTY_REF,
        target: EMPTY_REF,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    };
    let mut devices = vec![blank; count];
    discovery.devices(targets[0].reference, &mut devices)?;
    for descriptor in devices {
        if policy
            .device
            .is_some_and(|ordinal| ordinal != descriptor.reference.id)
        {
            continue;
        }
        // MLX doesn't report neutral physical identity/capacity here. Preserve Unknown;
        // the prepared backend still checks exact capabilities/schema/numerical requirements.
        let score = super::super::selection::score_candidate(
            policy,
            descriptor,
            None,
            Some(kernel),
            || discovery.device_facts(descriptor.reference),
        )?;
        output.push(Candidate {
            provider: Provider::Mlx,
            device: descriptor.reference,
            score,
        });
    }
    Ok(())
}

/// Publishes only terminal private MLX results; existing immutable owners are never overwritten
/// by device work. An exclusive Rust borrow permits replacement after successful completion.
pub(super) fn call_mlx_arguments<const N: usize>(
    invocation: &mut MlxInvocation,
    arguments: [super::super::arguments::PcuCallArgument<'_>; N],
) -> Result<(), PcuExecutionError> {
    match &mut invocation.kernel {
        MlxKernel::Conversion(kernel) => {
            conversion::call(kernel, invocation.prefixes[0].as_ref(), arguments)
        }
        MlxKernel::Transport { kernel, readback } => {
            transport::call(kernel, &invocation.prefixes, readback, arguments)
        }
        MlxKernel::Carrier(kernel) => {
            call_single_input_arguments(kernel, invocation.prefixes[0].as_ref(), arguments)
        }
        MlxKernel::Unary(kernel) => {
            call_single_input_arguments(kernel, invocation.prefixes[0].as_ref(), arguments)
        }
        MlxKernel::Binary {
            kernel,
            declarations,
            declaration_count,
        } => binary::call(
            kernel,
            &declarations[..*declaration_count],
            invocation.prefixes[0].as_ref(),
            arguments,
        ),
        MlxKernel::Integer {
            kernel,
            declarations,
            declaration_count,
        } => binary::call(
            kernel,
            &declarations[..*declaration_count],
            invocation.prefixes[0].as_ref(),
            arguments,
        ),
        MlxKernel::DivRemRoles {
            kernel,
            declarations,
            declaration_count,
            readback,
        } => div_rem::call(
            kernel,
            &declarations[..*declaration_count],
            &invocation.prefixes,
            readback,
            arguments,
        ),
        MlxKernel::DivRem { kernel, readback } => {
            let declarations = *kernel.input_bindings();
            div_rem::call(
                kernel,
                &declarations,
                &invocation.prefixes,
                readback,
                arguments,
            )
        }
    }
}

fn call_single_input_arguments<K: SingleInputKernel, const N: usize>(
    prepared: &mut K,
    prefix: Option<&MlxPreparedEncodedPrefix>,
    arguments: [super::super::arguments::PcuCallArgument<'_>; N],
) -> Result<(), PcuExecutionError> {
    use super::super::arguments::PcuCallArgumentKind;
    let (input, output) = collect_arguments(prepared, arguments)?;
    let (input, mut output) = match (input, output) {
        (PcuCallArgumentKind::Host(input), PcuCallArgumentKind::Host(output)) => {
            return prepared.call(&mut [input, output]).map_err(map_mlx_error);
        }
        pair => pair,
    };
    // Mixed/resident paths preflight both roles before staging, launching or changing a Rust owner.
    // No typed slices are reconstructed from backend bytes.
    let scalar = prepared.scalar_type();
    validate_output(
        &output,
        scalar,
        prepared.output_byte_len(),
        prepared.output_binding(),
        prefix,
    )?;
    let completion = match &input {
        #[cfg(all(feature = "cpu", feature = "tensor"))]
        PcuCallArgumentKind::CpuOwner(_) => {
            return Err(PcuExecutionError::Argument(
                super::super::PcuArgumentError::SessionMismatch,
            ));
        }
        PcuCallArgumentKind::Host(argument) => {
            validate_host(
                argument,
                scalar,
                crate::PcuBindingAccess::ReadOnly,
                prepared.input_byte_len(),
            )?;
            prepared.execute_encoded_bytes(scalar, argument.bytes())
        }
        PcuCallArgumentKind::MlxRead(argument) => prepared.execute_resident(argument.array),
        PcuCallArgumentKind::MlxWrite(_) => {
            return Err(map_mlx_error(crate::PcuHostDispatchError::AccessMismatch(
                prepared.input_binding(),
            )));
        }
        #[cfg(all(feature = "vulkan", feature = "tensor"))]
        PcuCallArgumentKind::VulkanRead(_) | PcuCallArgumentKind::VulkanWrite(_) => {
            return Err(PcuExecutionError::Argument(
                super::super::PcuArgumentError::SessionMismatch,
            ));
        }
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        PcuCallArgumentKind::ResidentRead(_) | PcuCallArgumentKind::ResidentWrite(_) => {
            return Err(PcuExecutionError::Argument(
                super::super::PcuArgumentError::SessionMismatch,
            ));
        }
    }
    .map_err(map_mlx_execution)?;
    let (completed, recovered) = completion.into_parts();
    publish_completed(
        &mut output,
        completed,
        scalar,
        prepared.output_byte_len(),
        prefix,
    )?;
    recovered.map_or(Ok(()), |fault| {
        Err(PcuExecutionError::ArithmeticFault(fault))
    })
}

fn validate_output(
    output: &super::super::arguments::PcuCallArgumentKind<'_>,
    scalar: crate::PcuScalarType,
    bytes: usize,
    binding: crate::PcuBindingRef,
    prefix: Option<&MlxPreparedEncodedPrefix>,
) -> Result<(), PcuExecutionError> {
    use super::super::arguments::PcuCallArgumentKind;
    match output {
        #[cfg(all(feature = "cpu", feature = "tensor"))]
        PcuCallArgumentKind::CpuOwner(_) => {
            return Err(PcuExecutionError::Argument(
                super::super::PcuArgumentError::SessionMismatch,
            ));
        }
        PcuCallArgumentKind::Host(argument) => {
            validate_host(argument, scalar, crate::PcuBindingAccess::ReadWrite, bytes)?;
        }
        PcuCallArgumentKind::MlxWrite(argument) => {
            argument
                .array
                .validate_access_available()
                .map_err(PcuExecutionError::MlxExecution)?;
            if argument.array.scalar_type() != scalar
                || argument.array.byte_len() < bytes
                || prefix
                    .is_some_and(|prefix| argument.array.element_count() != prefix.output_count())
                || (prefix.is_none() && argument.array.byte_len() != bytes)
            {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            }
        }
        PcuCallArgumentKind::MlxRead(_) => {
            return Err(map_mlx_error(crate::PcuHostDispatchError::AccessMismatch(
                binding,
            )));
        }
        #[cfg(all(feature = "vulkan", feature = "tensor"))]
        PcuCallArgumentKind::VulkanRead(_) | PcuCallArgumentKind::VulkanWrite(_) => {
            return Err(PcuExecutionError::Argument(
                super::super::PcuArgumentError::SessionMismatch,
            ));
        }
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        PcuCallArgumentKind::ResidentRead(_) | PcuCallArgumentKind::ResidentWrite(_) => {
            return Err(PcuExecutionError::Argument(
                super::super::PcuArgumentError::SessionMismatch,
            ));
        }
    }
    Ok(())
}

fn publish_completed(
    output: &mut super::super::arguments::PcuCallArgumentKind<'_>,
    completed: fusion_pcu_mlx::MlxEncodedArray,
    scalar: crate::PcuScalarType,
    bytes: usize,
    prefix: Option<&MlxPreparedEncodedPrefix>,
) -> Result<(), PcuExecutionError> {
    use super::super::arguments::PcuCallArgumentKind;
    match output {
        #[cfg(all(feature = "cpu", feature = "tensor"))]
        PcuCallArgumentKind::CpuOwner(_) => {
            return Err(PcuExecutionError::Argument(
                super::super::PcuArgumentError::SessionMismatch,
            ));
        }
        PcuCallArgumentKind::Host(argument) => completed
            .read_bytes_into(
                &mut argument
                    .bytes_mut()
                    .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?[..bytes],
            )
            .map_err(map_mlx_execution)?,
        PcuCallArgumentKind::MlxWrite(argument) => {
            // Exact session identity is checked before publication, even for host-input calls.
            if !completed.same_session(&argument.root.session)
                || completed.scalar_type() != scalar
                || completed.byte_len() != bytes
            {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            }
            // No host materialization or mutation of the previous native backing. The
            // full-length replacement is terminal before releasing the old Rust owner.
            let replacement = if let Some(prefix) = prefix {
                prefix
                    .execute(&completed, argument.array)
                    .map_err(map_mlx_execution)?
            } else {
                completed
            };
            *argument.array = replacement;
        }
        PcuCallArgumentKind::MlxRead(_) => return Err(PcuExecutionError::InvalidTensorSourcePlan),
        #[cfg(all(feature = "vulkan", feature = "tensor"))]
        PcuCallArgumentKind::VulkanRead(_) | PcuCallArgumentKind::VulkanWrite(_) => {
            return Err(PcuExecutionError::Argument(
                super::super::PcuArgumentError::SessionMismatch,
            ));
        }
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        PcuCallArgumentKind::ResidentRead(_) | PcuCallArgumentKind::ResidentWrite(_) => {
            return Err(PcuExecutionError::Argument(
                super::super::PcuArgumentError::SessionMismatch,
            ));
        }
    }
    Ok(())
}

fn validate_host(
    argument: &crate::PcuHostArgument<'_>,
    scalar: crate::PcuScalarType,
    access: crate::PcuBindingAccess,
    bytes: usize,
) -> Result<(), PcuExecutionError> {
    if argument.scalar() != scalar {
        return Err(map_mlx_error(crate::PcuHostDispatchError::TypeMismatch(
            argument.target(),
        )));
    }
    if argument.access() != access {
        return Err(map_mlx_error(crate::PcuHostDispatchError::AccessMismatch(
            argument.target(),
        )));
    }
    if argument.bytes().len() < bytes {
        return Err(map_mlx_error(crate::PcuHostDispatchError::BufferTooSmall(
            argument.target(),
        )));
    }
    Ok(())
}

fn collect_arguments<'a, K: SingleInputKernel, const N: usize>(
    prepared: &K,
    arguments: [super::super::arguments::PcuCallArgument<'a>; N],
) -> Result<
    (
        super::super::arguments::PcuCallArgumentKind<'a>,
        super::super::arguments::PcuCallArgumentKind<'a>,
    ),
    PcuExecutionError,
> {
    single_input::collect(
        prepared.input_binding(),
        prepared.output_binding(),
        prepared.scalar_type(),
        prepared.unused_bindings(),
        arguments,
    )
}

fn map_mlx_execution(error: MlxError) -> PcuExecutionError {
    match error {
        MlxError::Arithmetic(fault) => PcuExecutionError::ArithmeticFault(fault),
        other => PcuExecutionError::MlxExecution(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifted_arithmetic_preserves_recovery_and_binding_errors_keep_their_identity() {
        let fault = crate::PcuExecutionFault {
            invocation_id: 17,
            kind: crate::PcuExecutionFaultKind::ArithmeticUnderflow,
            recovered: true,
        };
        let error = map_mlx_error(crate::PcuHostDispatchError::Backend(MlxError::Arithmetic(
            fault,
        )));
        assert_eq!(error.recovered_range_fault(), Some(fault));
        let target = crate::PcuBindingRef::new(2, 7);
        assert!(matches!(
            map_mlx_error(crate::PcuHostDispatchError::TypeMismatch(target)),
            PcuExecutionError::MlxHostExecution(crate::PcuHostDispatchError::TypeMismatch(actual))
                if actual == target
        ));
    }
}

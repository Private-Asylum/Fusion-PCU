//! Typed resident calls; only private diagnostic records cross the host boundary.

#[rustfmt::skip]
use core::num::NonZeroU32;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingType,
    PcuDeviceArgument,
    PcuDeviceKernelBackend,
    PcuDispatchKernelIr,
    PcuDispatchSubmission,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuOwnedDispatchBackend,
    PcuOwnedDispatchBindingError,
    PcuOwnedDispatchMemorySession,
    PcuPreparedDeviceKernel,
    PcuValueType,
};
#[rustfmt::skip]
use super::{
    MetalOwnedDispatchBackend,
    MetalOwnedDispatchError,
    MetalPreparedDispatch,
};
#[rustfmt::skip]
use crate::{
    MetalError,
    MetalMemoryResource,
};

/// Resident checked executable with no implicit input upload or output publication transfer.
///
/// The GPU writes directly to the exclusive output allocation. Arithmetic failure can change its
/// prefix, as permitted by the neutral device call contract. Every call resets fresh private
/// records and reads those diagnostics only after terminal completion. Uncertain completion
/// quarantines the retained native session and its submission resources.
pub struct MetalPreparedDeviceKernel {
    dispatch: MetalPreparedDispatch,
}

impl PcuDeviceKernelBackend for MetalOwnedDispatchBackend {
    type Resource = MetalMemoryResource;
    type Prepared = MetalPreparedDeviceKernel;
    type Error = MetalOwnedDispatchError;
    fn prepare_device_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let [width, height, depth] = kernel.entry.logical_shape;
        let width = NonZeroU32::new(width).ok_or(MetalError::InvalidExtent)?;
        if height != 1 || depth != 1 {
            return Err(MetalError::Unsupported.into());
        }
        Ok(MetalPreparedDeviceKernel {
            dispatch: self.prepare_dispatch_owned_direct(
                PcuDispatchSubmission {
                    kernel,
                    shape: PcuInvocationShape::invocations(width),
                },
                PcuInvocationParameters::empty(),
            )?,
        })
    }
}
impl PcuPreparedDeviceKernel for MetalPreparedDeviceKernel {
    type Resource = MetalMemoryResource;
    type Error = MetalOwnedDispatchError;
    fn call(
        &mut self,
        arguments: &mut [PcuDeviceArgument<'_, MetalMemoryResource>],
    ) -> Result<(), Self::Error> {
        let bindings = self.bind_device_arguments(arguments)?;
        self.dispatch.validate_resources(&bindings)?;
        if let super::Program::DivRem(kernel) = &self.dispatch.program {
            return match MetalPreparedDispatch::execute_div_rem(kernel, &bindings)? {
                fusion_pcu::PcuCompletionOutcome::Succeeded => Ok(()),
                fusion_pcu::PcuCompletionOutcome::Fault(fault) => {
                    Err(MetalError::Arithmetic(fault).into())
                }
                fusion_pcu::PcuCompletionOutcome::Failed => Err(MetalError::Unsupported.into()),
            };
        }
        if let super::Program::DivRemRoles(kernel) = &self.dispatch.program {
            return match MetalPreparedDispatch::execute_div_rem_roles(kernel, &bindings)? {
                fusion_pcu::PcuCompletionOutcome::Succeeded => Ok(()),
                fusion_pcu::PcuCompletionOutcome::Fault(fault) => {
                    Err(MetalError::Arithmetic(fault).into())
                }
                fusion_pcu::PcuCompletionOutcome::Failed => Err(MetalError::Unsupported.into()),
            };
        }
        if let super::Program::Composed(kernel) = &self.dispatch.program {
            return match MetalPreparedDispatch::execute_composed(kernel, &bindings)? {
                fusion_pcu::PcuCompletionOutcome::Succeeded => Ok(()),
                fusion_pcu::PcuCompletionOutcome::Fault(fault) => {
                    Err(MetalError::Arithmetic(fault).into())
                }
                fusion_pcu::PcuCompletionOutcome::Failed => Err(MetalError::Unsupported.into()),
            };
        }
        if let super::Program::Transport(kernel) = &self.dispatch.program {
            return MetalPreparedDispatch::execute_transport(kernel, &bindings).map(|_| ());
        }
        let find = |target| {
            bindings
                .iter()
                .find(|binding| binding.target == target)
                .ok_or(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::Missing(target),
                ))
        };
        let inputs = self
            .dispatch
            .program
            .inputs()
            .ok_or(MetalError::Unsupported)?;
        let left = find(inputs[0])?;
        let right = find(inputs[1])?;
        let output = find(
            self.dispatch
                .program
                .output()
                .ok_or(MetalError::Unsupported)?,
        )?;
        // Avoid a RefCell conflict and prevent accidental aliasing through advanced resource owners.
        if std::rc::Rc::ptr_eq(&output.resource.buffer, &left.resource.buffer)
            || std::rc::Rc::ptr_eq(&output.resource.buffer, &right.resource.buffer)
        {
            return Err(MetalOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::UnsupportedLayout(output.target),
            ));
        }
        self.dispatch
            .program
            .execute_into(
                [
                    &left.resource.buffer.borrow(),
                    &right.resource.buffer.borrow(),
                ],
                &output.resource.buffer.borrow_mut(),
            )
            .map_err(MetalOwnedDispatchError::Metal)
    }
}

impl MetalOwnedDispatchBackend {
    /// Uploads an explicitly transferred22 byte-addressed sealed scalar buffer through shared memory services.
    ///
    /// # Errors
    /// Rejects unsupported types, zero extents, byte order, allocation or transfer failures.
    pub fn upload_buffer<T: fusion_pcu::PcuScalar>(
        &self,
        pool: fusion_pcu::PcuMemoryPoolId,
        values: &[T],
    ) -> Result<fusion_pcu::PcuDeviceBuffer<T, MetalMemoryResource>, MetalOwnedDispatchError> {
        use fusion_pcu::PcuMemoryProvider;
        if cfg!(target_endian = "big") || T::TYPE.bit_width() < 8 {
            return Err(MetalError::Unsupported.into());
        }
        if values.is_empty() {
            return Err(MetalError::InvalidExtent.into());
        }
        let host = fusion_pcu::PcuHostArgument::read(fusion_pcu::PcuBindingRef::new(0, 0), values);
        let mut memory = self.memory_provider(pool);
        let mut resource = memory
            .allocate(fusion_pcu::PcuMemoryAllocationRequest {
                pool,
                size_bytes: u64::try_from(host.bytes().len())
                    .map_err(|_| MetalError::InvalidExtent)?,
                alignment_bytes: 4,
                access: fusion_pcu::PcuMemoryAccess::ReadWrite,
                host_access: fusion_pcu::PcuMemoryHostAccess::TransferOnly,
                require_device_local: false,
            })
            .map_err(MetalOwnedDispatchError::Memory)?;
        memory
            .transfer_to(&mut resource, 0, host.bytes())
            .map_err(MetalOwnedDispatchError::Memory)?;
        Ok(fusion_pcu::PcuDeviceBuffer::new(resource, values.len()))
    }
    /// Explicitly transfers all typed buffer elements after terminal completion.
    ///
    /// # Errors
    /// Rejects mismatched lengths, byte order, resource affinity, or unavailable completion.
    pub fn download_buffer<T: fusion_pcu::PcuScalar>(
        &self,
        pool: fusion_pcu::PcuMemoryPoolId,
        buffer: &fusion_pcu::PcuDeviceBuffer<T, MetalMemoryResource>,
        values: &mut [T],
    ) -> Result<(), MetalOwnedDispatchError> {
        use fusion_pcu::PcuMemoryProvider;
        if cfg!(target_endian = "big") || values.len() != buffer.len() {
            return Err(MetalError::InvalidExtent.into());
        }
        let mut host =
            fusion_pcu::PcuHostArgument::read_write(fusion_pcu::PcuBindingRef::new(0, 0), values);
        self.memory_provider(pool)
            .transfer_from(
                buffer.resource(),
                0,
                host.bytes_mut().ok_or(MetalError::Unsupported)?,
            )
            .map_err(MetalOwnedDispatchError::Memory)
    }
}

const fn scalar_value_type(scalar: fusion_pcu::PcuScalarType) -> Option<PcuValueType> {
    if scalar.bit_width() >= 8 {
        Some(PcuValueType::Scalar(scalar))
    } else {
        None
    }
}

impl MetalPreparedDeviceKernel {
    fn bind_device_arguments(
        &self,
        arguments: &[PcuDeviceArgument<'_, MetalMemoryResource>],
    ) -> Result<Vec<fusion_pcu::PcuOwnedBinding<super::MetalOwnedResource>>, MetalOwnedDispatchError>
    {
        if let super::Program::Composed(kernel) = &self.dispatch.program {
            super::composed::validate_declarations(&kernel.borrow(), arguments)?;
        }
        if let super::Program::Transport(kernel) = &self.dispatch.program {
            super::transport::validate_declarations(kernel, arguments)?;
        }
        let bindings = arguments
            .iter()
            .enumerate()
            .filter(|(_, argument)| {
                !self
                    .dispatch
                    .program
                    .is_unread_declaration(argument.target())
            })
            .map(|(index, argument)| {
                if arguments[..index]
                    .iter()
                    .any(|other| other.target() == argument.target())
                {
                    return Err(MetalOwnedDispatchError::Binding(
                        PcuOwnedDispatchBindingError::Duplicate(argument.target()),
                    ));
                }
                let bytes = u64::try_from(argument.elements())
                    .ok()
                    .and_then(|count| {
                        count.checked_mul(u64::from(argument.scalar().bit_width()) / 8)
                    })
                    .ok_or(MetalError::InvalidExtent)?;
                let requirement = self
                    .dispatch
                    .requirements
                    .iter()
                    .find(|required| required.target == argument.target())
                    .ok_or_else(|| {
                        MetalOwnedDispatchError::Binding(PcuOwnedDispatchBindingError::Unexpected(
                            argument.target(),
                        ))
                    })?;
                if bytes < requirement.min_required_bytes {
                    return Err(MetalOwnedDispatchError::Binding(
                        PcuOwnedDispatchBindingError::BufferTooSmall {
                            binding: argument.target(),
                            required: requirement.min_required_bytes,
                            available: bytes,
                        },
                    ));
                }
                let value = scalar_value_type(argument.scalar()).ok_or_else(|| {
                    MetalOwnedDispatchError::Binding(PcuOwnedDispatchBindingError::TypeMismatch(
                        argument.target(),
                    ))
                })?;
                // Reuse the same physical affinity, real size and resource-access validation as bind.
                let backend = MetalOwnedDispatchBackend::new(
                    self.dispatch.session.clone(),
                    self.dispatch.device,
                );
                let binding = backend.bind(
                    argument.target(),
                    argument.access(),
                    PcuBindingType::Value(value),
                    argument.resource(),
                )?;
                if bytes > binding.byte_len {
                    return Err(MetalOwnedDispatchError::Binding(
                        PcuOwnedDispatchBindingError::BufferTooSmall {
                            binding: argument.target(),
                            required: bytes,
                            available: binding.byte_len,
                        },
                    ));
                }
                Ok(binding)
            })
            .collect::<Result<Vec<_>, MetalOwnedDispatchError>>()?;
        Ok(bindings)
    }
}

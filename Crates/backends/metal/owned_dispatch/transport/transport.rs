//! Actual typed transport resource schemas and retained terminal writer guards.

#[rustfmt::skip]
use std::cell::{
    Ref,
    RefMut,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDeviceArgument,
    PcuOwnedBinding,
    PcuOwnedDispatchBindingError,
    PcuValueType,
};
#[rustfmt::skip]
use super::{
    MetalOwnedDispatchError,
    MetalOwnedResource,
    MetalPreparedDispatch,
    Rc,
};
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalMemoryResource,
    MetalPreparedTransportKernel,
};

enum Guard<'a> {
    Read(Ref<'a, MetalBuffer>),
    Write(RefMut<'a, MetalBuffer>),
}
impl Guard<'_> {
    fn buffer(&self) -> &MetalBuffer {
        match self {
            Self::Read(buffer) => buffer,
            Self::Write(buffer) => buffer,
        }
    }
}

impl MetalPreparedDispatch {
    pub(super) fn execute_transport(
        kernel: &MetalPreparedTransportKernel,
        bindings: &[PcuOwnedBinding<MetalOwnedResource>],
    ) -> Result<PcuCompletionOutcome, MetalOwnedDispatchError> {
        let resources = kernel.plan().resources();
        for (slot, role) in resources.iter().enumerate() {
            let current = find(bindings, role.binding)?;
            for prior in &resources[..slot] {
                let other = find(bindings, prior.binding)?;
                if (role.minimum_write_elements != 0 || prior.minimum_write_elements != 0)
                    && Rc::ptr_eq(&current.resource.buffer, &other.resource.buffer)
                {
                    return Err(layout(role.binding));
                }
            }
        }
        let mut guards: [Option<Guard<'_>>; 4] = core::array::from_fn(|_| None);
        for (slot, role) in resources.iter().enumerate() {
            let lease = &find(bindings, role.binding)?.resource.buffer;
            guards[slot] = Some(if role.minimum_write_elements != 0 {
                Guard::Write(lease.try_borrow_mut().map_err(|_| layout(role.binding))?)
            } else {
                Guard::Read(lease.try_borrow().map_err(|_| layout(role.binding))?)
            });
        }
        let references: [&MetalBuffer; 4] = core::array::from_fn(|slot| {
            guards[slot.min(resources.len() - 1)]
                .as_ref()
                .unwrap()
                .buffer()
        });
        kernel.execute_into(&references[..resources.len()])?;
        Ok(PcuCompletionOutcome::Succeeded)
    }
}

pub(super) fn validate_declarations(
    kernel: &MetalPreparedTransportKernel,
    arguments: &[PcuDeviceArgument<'_, MetalMemoryResource>],
) -> Result<(), MetalOwnedDispatchError> {
    for (index, argument) in arguments.iter().enumerate() {
        let target = argument.target();
        if arguments[..index]
            .iter()
            .any(|prior| prior.target() == target)
        {
            return Err(MetalOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::Duplicate(target),
            ));
        }
        let declaration = kernel
            .plan()
            .declared_bindings()
            .iter()
            .find(|binding| binding.0 == target)
            .ok_or(MetalOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::Unexpected(target),
            ))?;
        if argument.scalar() != kernel.plan().scalar_type() {
            return Err(MetalOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::TypeMismatch(target),
            ));
        }
        if argument.access() != declaration.1 {
            return Err(MetalOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::AccessMismatch(target),
            ));
        }
    }
    for declaration in kernel.plan().declared_bindings() {
        if !arguments
            .iter()
            .any(|argument| argument.target() == declaration.0)
        {
            return Err(MetalOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::Missing(declaration.0),
            ));
        }
    }
    Ok(())
}

pub(super) fn requirements(
    kernel: &MetalPreparedTransportKernel,
) -> Vec<fusion_pcu::PcuOwnedBindingRequirement> {
    kernel
        .plan()
        .resources()
        .iter()
        .map(|role| {
            let access = kernel
                .plan()
                .declared_bindings()
                .iter()
                .find(|binding| binding.0 == role.binding)
                .unwrap()
                .1;
            fusion_pcu::PcuOwnedBindingRequirement {
                target: role.binding,
                access,
                binding_type: PcuBindingType::Value(PcuValueType::Scalar(
                    kernel.plan().scalar_type(),
                )),
                min_required_bytes: u64::from(role.minimum_elements())
                    * u64::from(kernel.plan().scalar_type().bit_width() / 8),
            }
        })
        .collect()
}

fn find(
    bindings: &[PcuOwnedBinding<MetalOwnedResource>],
    target: PcuBindingRef,
) -> Result<&PcuOwnedBinding<MetalOwnedResource>, MetalOwnedDispatchError> {
    bindings
        .iter()
        .find(|binding| binding.target == target)
        .ok_or(MetalOwnedDispatchError::Binding(
            PcuOwnedDispatchBindingError::Missing(target),
        ))
}
const fn layout(target: PcuBindingRef) -> MetalOwnedDispatchError {
    MetalOwnedDispatchError::Binding(PcuOwnedDispatchBindingError::UnsupportedLayout(target))
}

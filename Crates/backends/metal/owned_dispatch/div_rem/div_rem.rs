//! Native two-output owned/device publication without hidden host payload materialization.
#[rustfmt::skip]
use super::{
    MetalPreparedDispatch,
    MetalOwnedResource,
    MetalOwnedDispatchError,
    Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuOwnedBinding,
    PcuCompletionOutcome,
    PcuOwnedDispatchBindingError,
};
#[rustfmt::skip]
use crate::{
    MetalPreparedDivRemHostKernel,
    MetalPreparedDivRemRoleHostKernel,
    MetalError,
};
impl MetalPreparedDispatch {
    pub(super) fn execute_div_rem_roles(
        kernel: &MetalPreparedDivRemRoleHostKernel,
        bindings: &[PcuOwnedBinding<MetalOwnedResource>],
    ) -> Result<PcuCompletionOutcome, MetalOwnedDispatchError> {
        let find = |target| {
            bindings
                .iter()
                .find(|binding| binding.target == target)
                .ok_or(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::Missing(target),
                ))
        };
        let output_bindings = kernel.output_bindings();
        let q = find(output_bindings[0])?;
        let r = find(output_bindings[1])?;
        for output in [q, r] {
            if bindings.iter().any(|other| {
                other.target != output.target
                    && Rc::ptr_eq(&other.resource.buffer, &output.resource.buffer)
            }) {
                return Err(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::UnsupportedLayout(output.target),
                ));
            }
        }
        let borrow = |target| {
            find(target)?.resource.buffer.try_borrow().map_err(|_| {
                MetalOwnedDispatchError::Binding(PcuOwnedDispatchBindingError::UnsupportedLayout(
                    target,
                ))
            })
        };
        let inputs = kernel.input_bindings();
        let first = borrow(inputs[0])?;
        let second = if inputs.len() == 2 {
            Some(borrow(inputs[1])?)
        } else {
            None
        };
        let q = borrow(output_bindings[0])?;
        let r = borrow(output_bindings[1])?;
        let result = second.as_ref().map_or_else(
            || kernel.execute_joint(&[&first], [&q, &r]),
            |second| kernel.execute_joint(&[&first, second], [&q, &r]),
        );
        match result {
            Ok(()) => Ok(PcuCompletionOutcome::Succeeded),
            Err(MetalError::Arithmetic(fault)) => Ok(PcuCompletionOutcome::Fault(fault)),
            Err(error) => Err(error.into()),
        }
    }
    pub(super) fn execute_div_rem(
        kernel: &MetalPreparedDivRemHostKernel,
        bindings: &[PcuOwnedBinding<MetalOwnedResource>],
    ) -> Result<PcuCompletionOutcome, MetalOwnedDispatchError> {
        let find = |target| {
            bindings
                .iter()
                .find(|binding| binding.target == target)
                .ok_or(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::Missing(target),
                ))
        };
        let inputs = kernel.input_bindings();
        let outputs = kernel.output_bindings();
        let left = find(inputs[0])?;
        let right = find(inputs[1])?;
        let q = find(outputs[0])?;
        let r = find(outputs[1])?;
        for output in [q, r] {
            for other in bindings {
                if other.target != output.target
                    && Rc::ptr_eq(&other.resource.buffer, &output.resource.buffer)
                {
                    return Err(MetalOwnedDispatchError::Binding(
                        PcuOwnedDispatchBindingError::UnsupportedLayout(output.target),
                    ));
                }
            }
        }
        match kernel.execute_joint(
            [
                &left.resource.buffer.borrow(),
                &right.resource.buffer.borrow(),
            ],
            [&q.resource.buffer.borrow(), &r.resource.buffer.borrow()],
        ) {
            Ok(()) => Ok(PcuCompletionOutcome::Succeeded),
            Err(MetalError::Arithmetic(fault)) => Ok(PcuCompletionOutcome::Fault(fault)),
            Err(error) => Err(error.into()),
        }
    }
}

//! All actual host inputs retain stack buffers and transactional zero-allocation publication.
#[rustfmt::skip]
use fusion_pcu::{
    PcuHostDispatchError,
};
#[rustfmt::skip]
use crate::{
    MetalError,
    MetalHostKernelError,
    MetalMixedHostArgument,
};
use super::MetalPreparedDivRemRoleHostKernel;

impl MetalPreparedDivRemRoleHostKernel {
    pub(super) fn call_host_mixed(
        &mut self,
        arguments: &mut [MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        let input_bindings = self.input_bindings();
        let output_bindings = self.output_bindings();
        let mut inputs = [None, None];
        let mut outputs = [None, None];
        for argument in arguments {
            let target = argument.descriptor().0;
            if self.plan.is_unread_declaration(target) {
                continue;
            }
            let MetalMixedHostArgument::Host(argument) = argument else {
                return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
            };
            if let Some(slot) = input_bindings.iter().position(|&binding| binding == target) {
                inputs[slot] = Some(argument.bytes());
            } else if let Some(slot) = output_bindings
                .iter()
                .position(|&binding| binding == target)
            {
                outputs[slot] = Some(
                    argument
                        .bytes_mut()
                        .ok_or(PcuHostDispatchError::AccessMismatch(target))?,
                );
            }
        }
        let mut staged = [None, None];
        for slot in 0..input_bindings.len() {
            let bytes = inputs[slot].ok_or(PcuHostDispatchError::Missing(input_bindings[slot]))?;
            staged[slot] = Some(
                self.control
                    .session()
                    .upload_bytes(&bytes[..self.plan.input_byte_lengths()[slot]])
                    .map_err(PcuHostDispatchError::Backend)?,
            );
        }
        let get = |slot: usize| {
            staged[slot]
                .as_ref()
                .ok_or(PcuHostDispatchError::Backend(MetalError::Unsupported))
        };
        let operands = self.plan.operand_inputs();
        let private = self
            .control
            .execute([get(operands[0])?, get(operands[1])?])
            .map_err(PcuHostDispatchError::Backend)?;
        let [Some(q), Some(r)] = outputs else {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        };
        private
            .read_pair_bytes(
                &mut q[..self.plan.byte_len()],
                &mut r[..self.plan.byte_len()],
            )
            .map_err(PcuHostDispatchError::Backend)?;
        self.may_have_written = true;
        Ok(())
    }
}

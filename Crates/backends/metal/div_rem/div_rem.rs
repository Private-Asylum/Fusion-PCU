//! Detached exact primitive quotient/remainder source adapter, with joint host publication.
#[path = "roles/roles.rs"]
pub mod roles;
#[rustfmt::skip]
use fusion_pcu::{PcuHostKernelBackend,PcuPreparedHostKernel,PcuDispatchKernelIr,PcuHostArgument,PcuHostDispatchError,
 PcuBindingRef,PcuBindingAccess,PcuBindingType,PcuValueType,PcuValueTypeCaps,PcuDispatchOp,PcuDispatchDataOp,PcuRangePolicy};
#[rustfmt::skip]
use crate::{MetalSession,MetalPreparedDivRemControl,MetalError,MetalHostKernelError};
/// Explicit exact-session adapter; coarse/global dispatch admission is qualified separately.
pub struct MetalDivRemHostBackend {
    session: MetalSession,
}
/// Frozen real four-binding interface; no IR or caller storage is retained.
pub struct MetalPreparedDivRemHostKernel {
    control: MetalPreparedDivRemControl,
    bindings: [PcuBindingRef; 4],
    operands: [usize; 2],
    bytes: usize,
    publication: crate::runtime::div_rem::publication::Publication,
    may_have_written: bool,
}
impl MetalSession {
    #[must_use]
    pub fn checked_div_rem_backend(&self) -> MetalDivRemHostBackend {
        MetalDivRemHostBackend {
            session: self.clone(),
        }
    }
}
impl MetalPreparedDivRemHostKernel {
    #[must_use]
    pub const fn bindings(&self) -> &[PcuBindingRef; 4] {
        &self.bindings
    }
    /// Actual checked SSA operand bindings, retained cold.
    #[must_use]
    pub const fn input_bindings(&self) -> [PcuBindingRef; 2] {
        [
            self.bindings[self.operands[0]],
            self.bindings[self.operands[1]],
        ]
    }
    /// Both actual quotient and remainder binding references.
    #[must_use]
    pub const fn output_bindings(&self) -> [PcuBindingRef; 2] {
        [self.bindings[2], self.bindings[3]]
    }
    pub(crate) fn execute_joint(
        &self,
        inputs: [&crate::MetalBuffer; 2],
        outputs: [&crate::MetalBuffer; 2],
    ) -> Result<(), MetalError> {
        let packed = self.control.execute(inputs)?;
        self.publication.execute(&packed, outputs)
    }
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        4
    }
    #[must_use]
    pub const fn scalar_type(&self) -> fusion_pcu::PcuScalarType {
        self.control.scalar_type()
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.control.element_count()
    }
}
impl PcuHostKernelBackend for MetalDivRemHostBackend {
    type Prepared = MetalPreparedDivRemHostKernel;
    type Error = MetalHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        crate::admission::require_scalar_numerics(kernel).map_err(PcuHostDispatchError::Backend)?;
        if kernel.numerical_requirements.range_policy != PcuRangePolicy::Reject {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        }
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|binding| binding.binding_type)
        else {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        };
        fusion_pcu::validate_integer_checked_div_rem_kernel(
            kernel,
            PcuValueType::Scalar(scalar),
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported))?;
        let (body, count) = if let [PcuDispatchOp::GridStrideLoop { body, extent }, _] = kernel.ops
        {
            (
                *body,
                usize::try_from(*extent)
                    .map_err(|_| PcuHostDispatchError::Backend(MetalError::InvalidExtent))?,
            )
        } else {
            (
                &kernel.ops[..kernel.ops.len() - 1],
                usize::try_from(kernel.entry.logical_shape[0])
                    .map_err(|_| PcuHostDispatchError::Backend(MetalError::InvalidExtent))?,
            )
        };
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: first,
                binding: first_binding,
                ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: second,
                binding: second_binding,
                ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { lhs, rhs, .. }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: quotient, ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: remainder, ..
            }),
        ] = body
        else {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        };
        let slot = |value| {
            if value == first {
                Ok(0)
            } else if value == second {
                Ok(1)
            } else {
                Err(PcuHostDispatchError::Backend(MetalError::Unsupported))
            }
        };
        let control = self
            .session
            .prepare_checked_div_rem_control(scalar, count)
            .map_err(PcuHostDispatchError::Backend)?;
        let bytes = count
            .checked_mul(usize::from(scalar.bit_width()) / 8)
            .ok_or(PcuHostDispatchError::Backend(MetalError::InvalidExtent))?;
        Ok(MetalPreparedDivRemHostKernel {
            control,
            bindings: [*first_binding, *second_binding, *quotient, *remainder],
            operands: [slot(lhs)?, slot(rhs)?],
            publication: crate::runtime::div_rem::publication::Publication::prepare(
                &self.session,
                bytes,
                count,
            )
            .map_err(PcuHostDispatchError::Backend)?,
            may_have_written: false,
            bytes,
        })
    }
}
impl PcuPreparedHostKernel for MetalPreparedDivRemHostKernel {
    type Error = MetalHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.may_have_written = false;
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..index]
                .iter()
                .any(|prior| prior.target() == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let Some(slot) = self.bindings.iter().position(|binding| *binding == target) else {
                return Err(PcuHostDispatchError::Unexpected(target));
            };
            if argument.scalar() != self.scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            let access = if slot < 2 {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::ReadWrite
            };
            if argument.access() != access {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            if argument.bytes().len() < self.bytes {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        for &binding in &self.bindings {
            if !arguments
                .iter()
                .any(|argument| argument.target() == binding)
            {
                return Err(PcuHostDispatchError::Missing(binding));
            }
        }
        let mut inputs = [None; 2];
        let mut outputs = [None, None];
        for argument in arguments {
            let target = argument.target();
            let slot = self
                .bindings
                .iter()
                .position(|binding| *binding == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if slot < 2 {
                inputs[slot] = Some(argument.bytes());
            } else {
                outputs[slot - 2] = Some(
                    argument
                        .bytes_mut()
                        .ok_or(PcuHostDispatchError::AccessMismatch(target))?,
                );
            }
        }
        let [Some(q), Some(r)] = outputs else {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        };
        self.control
            .call_bytes(
                inputs[self.operands[0]].ok_or(PcuHostDispatchError::Missing(
                    self.bindings[self.operands[0]],
                ))?,
                inputs[self.operands[1]].ok_or(PcuHostDispatchError::Missing(
                    self.bindings[self.operands[1]],
                ))?,
                q,
                r,
            )
            .map_err(PcuHostDispatchError::Backend)?;
        self.may_have_written = true;
        Ok(())
    }
}

#[path = "mixed/mixed.rs"]
mod mixed;

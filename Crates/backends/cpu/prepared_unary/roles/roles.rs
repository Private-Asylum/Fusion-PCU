//! Detached actual-role schema for normal checked unary maps with unread declarations.
//! Only two resources execute. Arbitrarily many typed readonly declarations are retained cold;
//! warm calls neither allocate nor inspect IR. The fixed two-declaration prepared type is unchanged.
use alloc::vec::Vec;
#[rustfmt::skip]
use fusion_pcu::{
    describe_checked_float_unary_map,
    PcuBindingAccess,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuPreparedHostKernel,
    PcuReproducibility,
};
#[rustfmt::skip]
use super::{freeze, normal_base, PcuCpuPreparedUnary};
#[rustfmt::skip]
use crate::{host, PcuCpuHostError};

/// Cold-owned declaration metadata and one statically selected checked executable.
#[derive(Debug, Clone)]
pub struct PcuCpuPreparedUnaryRoles {
    executable: PcuCpuPreparedUnary,
    schema: Vec<(PcuBindingRef, usize, PcuBindingAccess)>,
}
impl PcuCpuPreparedUnaryRoles {
    pub(crate) fn prepare<T: PcuCheckedFloat>(
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self, PcuCpuHostError> {
        if kernel.bindings.len() <= 2
            || kernel
                .numerical_requirements
                .numerical_options
                .reproducibility
                != PcuReproducibility::Unspecified
        {
            return Err(PcuCpuHostError::UnsupportedProfile);
        }
        host::validated_region(kernel)?;
        let description = describe_checked_float_unary_map(kernel)
            .map_err(|_| PcuCpuHostError::UnsupportedProfile)?;
        let executable = freeze::<T>(description, 17920 + normal_base(T::TYPE)? - 288)?;
        let mut schema = Vec::with_capacity(kernel.bindings.len());
        for binding in kernel.bindings {
            let target = binding.reference();
            let count = if target == executable.output {
                executable.extent
            } else if target == executable.input {
                if executable.broadcast {
                    1
                } else {
                    executable.extent
                }
            } else {
                if binding.access != PcuBindingAccess::ReadOnly {
                    return Err(PcuCpuHostError::UnsupportedProfile);
                }
                0
            };
            schema.push((target, count, binding.access));
        }
        Ok(Self { executable, schema })
    }
    #[must_use]
    pub const fn local_id(&self) -> u32 {
        self.executable.local_id()
    }
    #[must_use]
    pub const fn range_policy(&self) -> fusion_pcu::PcuRangePolicy {
        self.executable.range_policy()
    }
    #[must_use]
    pub const fn underflow_policy(&self) -> fusion_pcu::PcuFloatUnderflowPolicy {
        self.executable.underflow_policy()
    }
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.schema.len()
    }
}
impl PcuPreparedHostKernel for PcuCpuPreparedUnaryRoles {
    type Error = PcuCpuHostError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        host::validate_arguments(
            arguments,
            &self.schema,
            self.executable.scalar,
            self.executable.size,
        )
        .map_err(PcuCpuHostError::Arguments)?;
        self.executable.call_validated(arguments)
    }
}

//! Cold-frozen exact checked Neg/ReLU maps with descriptor-qualified Portable admission.
use core::marker::PhantomData;
#[path = "execution/execution.rs"]
mod execution;
#[cfg(any(feature = "std", feature = "tensor"))]
#[path = "roles/roles.rs"]
mod roles;
#[cfg(any(feature = "std", feature = "tensor"))]
pub use roles::PcuCpuPreparedUnaryRoles;
#[rustfmt::skip]
use fusion_pcu::{
    describe_portable_v1_unary_map,
    describe_checked_float_unary_map,
    PcuCheckedFloatUnaryMapDescription,
    PcuBindingAccess,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuDispatchFloatUnaryOp,
    PcuDispatchKernelIr,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{PcuCpuHostError,host};

/// Explicit checked scalar unary backend for six sealed float representations.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuCheckedUnary<T: PcuCheckedFloat> {
    marker: PhantomData<T>,
}
impl<T: PcuCheckedFloat> PcuCpuCheckedUnary<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}
impl<T: PcuCheckedFloat> Default for PcuCpuCheckedUnary<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Detached operation, policy, typed schema and function pointer; no IR/host borrow or allocation.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuPreparedUnary {
    input: PcuBindingRef,
    output: PcuBindingRef,
    extent: usize,
    broadcast: bool,
    scalar: PcuScalarType,
    size: usize,
    operation: PcuDispatchFloatUnaryOp,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    local_id: u32,
    execute: execution::Executable,
}
impl PcuCpuPreparedUnary {
    #[must_use]
    pub const fn operation(&self) -> PcuDispatchFloatUnaryOp {
        self.operation
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn underflow_policy(&self) -> PcuFloatUnderflowPolicy {
        self.underflow
    }
    #[must_use]
    pub const fn range_policy(&self) -> PcuRangePolicy {
        self.range
    }
    #[must_use]
    pub const fn local_id(&self) -> u32 {
        self.local_id
    }
    pub(super) const fn host_schema(&self) -> [(PcuBindingRef, usize, PcuBindingAccess); 3] {
        [
            (
                self.input,
                if self.broadcast { 1 } else { self.extent },
                PcuBindingAccess::ReadOnly,
            ),
            (self.output, self.extent, PcuBindingAccess::ReadWrite),
            (self.output, 0, PcuBindingAccess::ReadWrite),
        ]
    }
}
impl<T: PcuCheckedFloat> PcuHostKernelBackend for PcuCpuCheckedUnary<T> {
    type Prepared = PcuCpuPreparedUnary;
    type Error = PcuCpuHostError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let portable = kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            == PcuReproducibility::PortableV1;
        if portable {
            // Keep descriptor-first admission and the independently qualified two-declaration profile.
            describe_portable_v1_unary_map(kernel)
                .map_err(|_| PcuCpuHostError::UnsupportedProfile)?;
        }
        host::validated_region(kernel)?;
        if kernel.bindings.len() != 2 {
            return Err(PcuCpuHostError::UnsupportedProfile);
        }
        let description = describe_checked_float_unary_map(kernel)
            .map_err(|_| PcuCpuHostError::UnsupportedProfile)?;
        if !portable
            && matches!(T::TYPE, PcuScalarType::F32 | PcuScalarType::F64)
            && description.operation == PcuDispatchFloatUnaryOp::Neg
            && description.requirements.range_policy == PcuRangePolicy::Reject
            && !description.broadcast_input
        {
            // Preserve the existing canonical scalar/SIMD Reject Neg executor and IDs.
            return Err(PcuCpuHostError::UnsupportedProfile);
        }
        let normal_base = normal_base(T::TYPE)?;
        let base = if portable {
            16384 + normal_base - 288
        } else {
            normal_base
        };
        freeze::<T>(description, base)
    }
}

const fn normal_base(scalar: PcuScalarType) -> Result<u32, PcuCpuHostError> {
    match scalar {
        PcuScalarType::F16 => Ok(288),
        PcuScalarType::BF16 => Ok(292),
        PcuScalarType::F8E4M3FN => Ok(296),
        PcuScalarType::F8E5M2 => Ok(300),
        PcuScalarType::F32 => Ok(304),
        PcuScalarType::F64 => Ok(308),
        _ => Err(PcuCpuHostError::UnsupportedProfile),
    }
}

fn freeze<T: PcuCheckedFloat>(
    description: PcuCheckedFloatUnaryMapDescription,
    base: u32,
) -> Result<PcuCpuPreparedUnary, PcuCpuHostError> {
    if description.scalar != T::TYPE {
        return Err(PcuCpuHostError::UnsupportedProfile);
    }
    let operation = description.operation;
    let range = description.requirements.range_policy;
    Ok(PcuCpuPreparedUnary {
        input: description.input_binding,
        output: description.output_binding,
        extent: usize::try_from(description.logical_extent)
            .map_err(|_| PcuCpuHostError::UnsupportedProfile)?,
        broadcast: description.broadcast_input,
        scalar: T::TYPE,
        size: T::HOST_SIZE,
        operation,
        underflow: description.requirements.float_underflow,
        range,
        local_id: base
            + u32::from(operation == PcuDispatchFloatUnaryOp::Relu)
            + if range == PcuRangePolicy::Clamp { 2 } else { 0 },
        execute: execution::prepare::<T>(operation, range, description.broadcast_input)?,
    })
}

impl PcuPreparedHostKernel for PcuCpuPreparedUnary {
    type Error = PcuCpuHostError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        host::validate_arguments(arguments, &self.host_schema()[..2], self.scalar, self.size)
            .map_err(PcuCpuHostError::Arguments)?;
        self.call_validated(arguments)
    }
}
impl PcuCpuPreparedUnary {
    fn call_validated(&self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), PcuCpuHostError> {
        let source = arguments
            .iter()
            .position(|arg| arg.target() == self.input)
            .expect("validated input");
        let destination = arguments
            .iter()
            .position(|arg| arg.target() == self.output)
            .expect("validated output");
        let (input, output) = if source < destination {
            let (before, after) = arguments.split_at_mut(destination);
            (&before[source], &mut after[0])
        } else {
            let (before, after) = arguments.split_at_mut(source);
            (&after[0], &mut before[destination])
        };
        (self.execute)(
            input.bytes(),
            output.bytes_mut().expect("validated output access"),
            self.extent,
            self.underflow,
        )
    }
}

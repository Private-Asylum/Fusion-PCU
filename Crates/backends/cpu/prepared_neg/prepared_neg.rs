//! Owned preparation of the bounded checked F32/F64 sign-bit Neg profiles.

#[rustfmt::skip]
use fusion_pcu::{
    validate_checked_float_map_kernel,
    validate_typed_dispatch_value_flow,
    CheckedFloatMapValidationError,
    PcuTypedDispatchValidationError,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use crate::{
    PcuCpuImplementation,
    PcuCpuImplementationUnavailable,
    PcuCpuProcessor,
};

#[path = "execution/execution.rs"]
mod execution;

type NegExecution = fn(
    PcuCpuCheckedNeg,
    &[u8],
    &mut [u8],
    PcuFloatUnderflowPolicy,
) -> Result<(), PcuCpuPreparedNegError>;

/// Cold admission, host schema, or terminal checked Neg failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuPreparedNegError {
    UnsupportedProfile,
    HeaderUnderflowMismatch,
    InvalidKernel(CheckedFloatMapValidationError),
    InvalidValueFlow(PcuTypedDispatchValidationError),
    InvalidGridExtent,
    ImplementationUnavailable(PcuCpuImplementationUnavailable),
    InvalidArguments,
    /// All host outputs are preserved on a checked fault; the lowest logical invocation wins.
    Fault(PcuExecutionFault),
}

/// Explicit CPU backend for checked F32/F64 Neg. It is never installed as a fallback.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuCheckedNeg {
    processor: PcuCpuProcessor,
    implementation: PcuCpuImplementation,
}

impl PcuCpuCheckedNeg {
    /// Admits an explicit instruction choice outside the warm call.
    ///
    /// # Errors
    /// Rejects unavailable SIMD instead of silently choosing scalar execution.
    pub const fn new(
        processor: PcuCpuProcessor,
        implementation: PcuCpuImplementation,
    ) -> Result<Self, PcuCpuImplementationUnavailable> {
        match processor.require(implementation) {
            Ok(()) => Ok(Self {
                processor,
                implementation,
            }),
            Err(error) => Err(error),
        }
    }

    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            processor: PcuCpuProcessor::scalar(),
            implementation: PcuCpuImplementation::Scalar,
        }
    }
}

/// Detached executable/schema: owns no temporary IR or host borrow and allocates no storage.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuPreparedNeg {
    backend: PcuCpuCheckedNeg,
    input: PcuBindingRef,
    output: PcuBindingRef,
    extent: usize,
    underflow: PcuFloatUnderflowPolicy,
    scalar: PcuScalarType,
    element_size: usize,
    execute: NegExecution,
}

impl PcuCpuPreparedNeg {
    pub(super) const fn host_schema(&self) -> [(PcuBindingRef, usize, PcuBindingAccess); 3] {
        [
            (self.input, self.extent, PcuBindingAccess::ReadOnly),
            (self.output, self.extent, PcuBindingAccess::ReadWrite),
            (self.output, 0, PcuBindingAccess::ReadWrite),
        ]
    }
    #[must_use]
    pub const fn implementation(&self) -> PcuCpuImplementation {
        self.backend.implementation
    }

    #[must_use]
    pub const fn underflow_policy(&self) -> PcuFloatUnderflowPolicy {
        self.underflow
    }

    /// Concrete scalar representation frozen during preparation.
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }

    pub(super) const fn element_size(&self) -> usize {
        self.element_size
    }

    pub(super) const fn local_id(&self) -> u32 {
        let instruction = match self.implementation() {
            PcuCpuImplementation::Scalar => 0,
            PcuCpuImplementation::Sse2 => 1,
            PcuCpuImplementation::Avx2 => 2,
            PcuCpuImplementation::Neon => 3,
        };
        match self.scalar {
            PcuScalarType::F64 => 28 + instruction,
            _ => instruction,
        }
    }
}

impl PcuHostKernelBackend for PcuCpuCheckedNeg {
    type Prepared = PcuCpuPreparedNeg;
    type Error = PcuCpuPreparedNegError;

    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        self.processor
            .require(self.implementation)
            .map_err(PcuCpuPreparedNegError::ImplementationUnavailable)?;
        let scalar = validate_neg_profile(kernel)?;
        let (body, extent, index) = match kernel.ops {
            [
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (*body, *extent, PcuDispatchIndex::GridStrideId),
            [
                body @ ..,
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (
                body,
                kernel.entry.logical_shape[0],
                PcuDispatchIndex::InvocationId,
            ),
            _ => return Err(PcuCpuPreparedNegError::UnsupportedProfile),
        };
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: loaded,
                binding: input,
                index: load_index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                value_type,
                op: PcuDispatchFloatUnaryOp::Neg,
                underflow_policy,
                range_policy: PcuRangePolicy::Reject,
                result: negated,
                value,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output,
                index: store_index,
                value: stored,
            }),
        ] = body
        else {
            return Err(PcuCpuPreparedNegError::UnsupportedProfile);
        };
        if kernel.numerical_requirements.float_underflow != *underflow_policy {
            return Err(PcuCpuPreparedNegError::HeaderUnderflowMismatch);
        }
        if kernel.numerical_requirements.range_policy != PcuRangePolicy::Reject
            || *value_type != PcuValueType::Scalar(scalar)
            || loaded != value
            || negated != stored
            || load_index != &index
            || store_index != &index
            || input == output
            || kernel.bindings.len() != 2
        {
            return Err(PcuCpuPreparedNegError::UnsupportedProfile);
        }
        let input_schema = kernel
            .bindings
            .iter()
            .find(|binding| binding.reference() == *input);
        if !matches!(input_schema, Some(binding) if binding.access == PcuBindingAccess::ReadOnly) {
            return Err(PcuCpuPreparedNegError::UnsupportedProfile);
        }
        let (element_size, execute): (usize, NegExecution) = match scalar {
            PcuScalarType::F64 => (8, execution::f64),
            _ => (4, execution::f32),
        };
        Ok(PcuCpuPreparedNeg {
            backend: *self,
            input: *input,
            output: *output,
            extent: usize::try_from(extent)
                .map_err(|_| PcuCpuPreparedNegError::UnsupportedProfile)?,
            underflow: *underflow_policy,
            scalar,
            element_size,
            execute,
        })
    }
}

impl PcuPreparedHostKernel for PcuCpuPreparedNeg {
    type Error = PcuCpuPreparedNegError;

    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        if arguments.len() != 2 {
            return Err(PcuCpuPreparedNegError::InvalidArguments);
        }
        let source = arguments
            .iter()
            .position(|argument| argument.target() == self.input)
            .ok_or(PcuCpuPreparedNegError::InvalidArguments)?;
        let destination = arguments
            .iter()
            .position(|argument| argument.target() == self.output)
            .ok_or(PcuCpuPreparedNegError::InvalidArguments)?;
        let bytes = self
            .extent
            .checked_mul(self.element_size)
            .ok_or(PcuCpuPreparedNegError::InvalidArguments)?;
        if source == destination
            || arguments
                .iter()
                .any(|argument| argument.scalar() != self.scalar || argument.bytes().len() < bytes)
            || arguments[source].access() != PcuBindingAccess::ReadOnly
            || arguments[destination].access() != PcuBindingAccess::ReadWrite
        {
            return Err(PcuCpuPreparedNegError::InvalidArguments);
        }
        let (input, output) = if source < destination {
            let (before, after) = arguments.split_at_mut(destination);
            (&before[source], &mut after[0])
        } else {
            let (before, after) = arguments.split_at_mut(source);
            (&after[0], &mut before[destination])
        };
        (self.execute)(
            self.backend,
            &input.bytes()[..bytes],
            &mut output.bytes_mut().expect("validated mutable argument")[..bytes],
            self.underflow,
        )
    }
}

fn validate_neg_profile(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuScalarType, PcuCpuPreparedNegError> {
    // Exact scalar arithmetic does not certify the complete PortableV1 contract.
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == fusion_pcu::PcuReproducibility::PortableV1
    {
        return Err(PcuCpuPreparedNegError::UnsupportedProfile);
    }
    if kernel.entry.logical_shape.contains(&0) {
        return Err(PcuCpuPreparedNegError::InvalidKernel(
            CheckedFloatMapValidationError::InvalidLogicalShape,
        ));
    }
    if kernel.entry.logical_shape[1..] != [1, 1] {
        return Err(PcuCpuPreparedNegError::UnsupportedProfile);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == binding.reference())
        {
            return Err(PcuCpuPreparedNegError::InvalidKernel(
                CheckedFloatMapValidationError::DuplicateBinding(binding.reference()),
            ));
        }
    }
    if matches!(
        kernel.ops,
        [PcuDispatchOp::GridStrideLoop { extent: 0, .. }, ..]
    ) {
        return Err(PcuCpuPreparedNegError::InvalidGridExtent);
    }
    validate_typed_dispatch_value_flow(kernel).map_err(PcuCpuPreparedNegError::InvalidValueFlow)?;
    if kernel.bindings.len() != 2 {
        return Err(PcuCpuPreparedNegError::UnsupportedProfile);
    }
    // A different, otherwise well-typed scalar operation is valid unsupported work.
    let region = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { body, .. },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => *body,
        ops => ops,
    };
    let scalar =
        region
            .iter()
            .find_map(|op| match op {
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                    value_type:
                        PcuValueType::Scalar(scalar @ (PcuScalarType::F32 | PcuScalarType::F64)),
                    op: PcuDispatchFloatUnaryOp::Neg,
                    ..
                }) => Some(*scalar),
                _ => None,
            })
            .ok_or(PcuCpuPreparedNegError::UnsupportedProfile)?;
    match validate_checked_float_map_kernel(
        kernel,
        PcuValueType::Scalar(scalar),
        PcuValueTypeCaps::for_scalar(scalar),
    ) {
        Ok(()) => {}
        Err(
            CheckedFloatMapValidationError::UnsupportedType
            | CheckedFloatMapValidationError::UnsupportedInterface
            | CheckedFloatMapValidationError::UnsupportedRequirements
            | CheckedFloatMapValidationError::UnsupportedOperation(_),
        ) => return Err(PcuCpuPreparedNegError::UnsupportedProfile),
        Err(error) => return Err(PcuCpuPreparedNegError::InvalidKernel(error)),
    }
    Ok(scalar)
}

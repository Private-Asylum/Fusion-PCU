//! Static cold selection of the admitted CPU host profiles.

#[rustfmt::skip]
use fusion_pcu::{
    validate_typed_dispatch_value_flow,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchKernelIr,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuExecutionFault,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuScalarType,
    PcuTypedDispatchValidationError,
    PcuValueType,
};
#[rustfmt::skip]
use crate::{
    PcuCpuCheckedInteger,
    PcuCpuCheckedIntegerError,
    PcuCpuCheckedNeg,
    PcuCpuImplementation,
    PcuCpuImplementationUnavailable,
    PcuCpuPreparedInteger,
    PcuCpuPreparedNeg,
    PcuCpuPreparedNegError,
    PcuCpuProcessor,
};

/// A complete, typed host-argument rejection before execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuHostArgumentError {
    Count {
        expected: usize,
        actual: usize,
    },
    DuplicateBinding(PcuBindingRef),
    MissingBinding(PcuBindingRef),
    TypeMismatch {
        binding: PcuBindingRef,
        expected: PcuScalarType,
        actual: PcuScalarType,
    },
    AccessMismatch {
        binding: PcuBindingRef,
        expected: PcuBindingAccess,
        actual: PcuBindingAccess,
    },
    BufferTooSmall {
        binding: PcuBindingRef,
        required_bytes: usize,
        actual_bytes: usize,
    },
    ExtentOverflow(PcuBindingRef),
}

/// Unified CPU errors retain exact typed-provider and host-schema diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuHostError {
    UnsupportedProfile,
    InvalidLogicalShape([u32; 3]),
    DuplicateBinding(PcuBindingRef),
    InvalidValueFlow(PcuTypedDispatchValidationError),
    MissingReturn,
    InvalidGridExtent,
    InvalidBindingAccess {
        binding: PcuBindingRef,
        access: PcuBindingAccess,
        required: PcuBindingAccess,
    },
    Arguments(PcuCpuHostArgumentError),
    Neg(PcuCpuPreparedNegError),
    Integer(PcuCpuCheckedIntegerError),
}

impl PcuCpuHostError {
    #[must_use]
    pub const fn fault(self) -> Option<PcuExecutionFault> {
        match self {
            Self::Neg(PcuCpuPreparedNegError::Fault(fault))
            | Self::Integer(PcuCpuCheckedIntegerError::Fault(fault)) => Some(fault),
            _ => None,
        }
    }
}

/// Explicit CPU host backend; instruction choice is resolved only during construction/preparation.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuHostBackend {
    neg: PcuCpuCheckedNeg,
}

impl PcuCpuHostBackend {
    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            neg: PcuCpuCheckedNeg::scalar(),
        }
    }

    /// Constructs a backend with an explicit, already detected Neg instruction implementation.
    ///
    /// # Errors
    /// Rejects unavailable instructions. Integer arithmetic remains scalar.
    pub const fn new(
        processor: PcuCpuProcessor,
        implementation: PcuCpuImplementation,
    ) -> Result<Self, PcuCpuImplementationUnavailable> {
        match PcuCpuCheckedNeg::new(processor, implementation) {
            Ok(neg) => Ok(Self { neg }),
            Err(error) => Err(error),
        }
    }

    /// Detects available CPU instructions, retaining explicit scalar integer execution.
    ///
    /// # Panics
    /// Panics only if the internal detected-implementation admission invariant is broken.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn detect() -> Self {
        let processor = PcuCpuProcessor::detect();
        Self::new(processor, processor.widest()).expect("detected instruction choice")
    }
}

/// Detached static prepared CPU variants; no trait-object registry or warm IR classification.
#[derive(Debug, Clone, Copy)]
pub enum PcuCpuPreparedHost {
    Neg(PcuCpuPreparedNeg),
    I8(PcuCpuPreparedInteger<i8>),
    U8(PcuCpuPreparedInteger<u8>),
    I16(PcuCpuPreparedInteger<i16>),
    U16(PcuCpuPreparedInteger<u16>),
    I32(PcuCpuPreparedInteger<i32>),
    U32(PcuCpuPreparedInteger<u32>),
    I64(PcuCpuPreparedInteger<i64>),
    U64(PcuCpuPreparedInteger<u64>),
}

impl PcuHostKernelBackend for PcuCpuHostBackend {
    type Prepared = PcuCpuPreparedHost;
    type Error = PcuCpuHostError;

    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let body = validated_region(kernel)?;
        let mut operations = body.iter().filter_map(|op| match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                value_type,
                op: PcuDispatchFloatUnaryOp::Neg,
                ..
            }) => Some((*value_type, true)),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { value_type, .. }) => {
                Some((*value_type, false))
            }
            _ => None,
        });
        let Some((value_type, neg)) = operations.next() else {
            return Err(PcuCpuHostError::UnsupportedProfile);
        };
        if operations.next().is_some() {
            return Err(PcuCpuHostError::UnsupportedProfile);
        }
        if neg {
            return match value_type {
                PcuValueType::Scalar(PcuScalarType::F32 | PcuScalarType::F64) => self
                    .neg
                    .prepare_host_kernel(kernel)
                    .map(PcuCpuPreparedHost::Neg)
                    .map_err(PcuCpuHostError::Neg),
                _ => Err(PcuCpuHostError::UnsupportedProfile),
            };
        }
        macro_rules! integer {
            ($ty:ty, $variant:ident) => {
                PcuCpuCheckedInteger::<$ty>::new()
                    .prepare_host_kernel(kernel)
                    .map(PcuCpuPreparedHost::$variant)
                    .map_err(PcuCpuHostError::Integer)
            };
        }
        match value_type {
            PcuValueType::Scalar(PcuScalarType::I8) => integer!(i8, I8),
            PcuValueType::Scalar(PcuScalarType::U8) => integer!(u8, U8),
            PcuValueType::Scalar(PcuScalarType::I16) => integer!(i16, I16),
            PcuValueType::Scalar(PcuScalarType::U16) => integer!(u16, U16),
            PcuValueType::Scalar(PcuScalarType::I32) => integer!(i32, I32),
            PcuValueType::Scalar(PcuScalarType::U32) => integer!(u32, U32),
            PcuValueType::Scalar(PcuScalarType::I64) => integer!(i64, I64),
            PcuValueType::Scalar(PcuScalarType::U64) => integer!(u64, U64),
            _ => Err(PcuCpuHostError::UnsupportedProfile),
        }
    }
}

impl PcuPreparedHostKernel for PcuCpuPreparedHost {
    type Error = PcuCpuHostError;

    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        let (schema, count, scalar, size) = match self {
            Self::Neg(prepared) => (
                prepared.host_schema(),
                2,
                prepared.scalar_type(),
                prepared.element_size(),
            ),
            Self::I8(prepared) => (prepared.host_schema(), 3, PcuScalarType::I8, 1),
            Self::U8(prepared) => (prepared.host_schema(), 3, PcuScalarType::U8, 1),
            Self::I16(prepared) => (prepared.host_schema(), 3, PcuScalarType::I16, 2),
            Self::U16(prepared) => (prepared.host_schema(), 3, PcuScalarType::U16, 2),
            Self::I32(prepared) => (prepared.host_schema(), 3, PcuScalarType::I32, 4),
            Self::U32(prepared) => (prepared.host_schema(), 3, PcuScalarType::U32, 4),
            Self::I64(prepared) => (prepared.host_schema(), 3, PcuScalarType::I64, 8),
            Self::U64(prepared) => (prepared.host_schema(), 3, PcuScalarType::U64, 8),
        };
        validate_arguments(arguments, &schema[..count], scalar, size)
            .map_err(PcuCpuHostError::Arguments)?;
        macro_rules! integer {
            ($prepared:ident) => {
                $prepared.call(arguments).map_err(PcuCpuHostError::Integer)
            };
        }
        match self {
            Self::Neg(prepared) => prepared.call(arguments).map_err(PcuCpuHostError::Neg),
            Self::I8(prepared) => integer!(prepared),
            Self::U8(prepared) => integer!(prepared),
            Self::I16(prepared) => integer!(prepared),
            Self::U16(prepared) => integer!(prepared),
            Self::I32(prepared) => integer!(prepared),
            Self::U32(prepared) => integer!(prepared),
            Self::I64(prepared) => integer!(prepared),
            Self::U64(prepared) => integer!(prepared),
        }
    }
}

fn validate_arguments(
    arguments: &[PcuHostArgument<'_>],
    schema: &[(PcuBindingRef, usize, PcuBindingAccess)],
    scalar: PcuScalarType,
    size: usize,
) -> Result<(), PcuCpuHostArgumentError> {
    if arguments.len() != schema.len() {
        return Err(PcuCpuHostArgumentError::Count {
            expected: schema.len(),
            actual: arguments.len(),
        });
    }
    for (index, argument) in arguments.iter().enumerate() {
        if arguments[..index]
            .iter()
            .any(|prior| prior.target() == argument.target())
        {
            return Err(PcuCpuHostArgumentError::DuplicateBinding(argument.target()));
        }
    }
    for &(binding, elements, expected_access) in schema {
        let argument = arguments
            .iter()
            .find(|argument| argument.target() == binding)
            .ok_or(PcuCpuHostArgumentError::MissingBinding(binding))?;
        if argument.scalar() != scalar {
            return Err(PcuCpuHostArgumentError::TypeMismatch {
                binding,
                expected: scalar,
                actual: argument.scalar(),
            });
        }
        if argument.access() != expected_access {
            return Err(PcuCpuHostArgumentError::AccessMismatch {
                binding,
                expected: expected_access,
                actual: argument.access(),
            });
        }
        let required_bytes = elements
            .checked_mul(size)
            .ok_or(PcuCpuHostArgumentError::ExtentOverflow(binding))?;
        if argument.bytes().len() < required_bytes {
            return Err(PcuCpuHostArgumentError::BufferTooSmall {
                binding,
                required_bytes,
                actual_bytes: argument.bytes().len(),
            });
        }
    }
    Ok(())
}

#[path = "offers/offers.rs"]
mod offers;
#[rustfmt::skip]
pub use offers::{
    PcuCpuHostOfferError,
    PcuCpuHostOffers,
};

fn validated_region<'a>(
    kernel: &PcuDispatchKernelIr<'a>,
) -> Result<&'a [PcuDispatchOp<'a>], PcuCpuHostError> {
    if kernel.entry.logical_shape.contains(&0) {
        return Err(PcuCpuHostError::InvalidLogicalShape(
            kernel.entry.logical_shape,
        ));
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == binding.reference())
        {
            return Err(PcuCpuHostError::DuplicateBinding(binding.reference()));
        }
    }
    if matches!(
        kernel.ops,
        [PcuDispatchOp::GridStrideLoop { extent: 0, .. }, ..]
    ) {
        return Err(PcuCpuHostError::InvalidGridExtent);
    }
    match validate_typed_dispatch_value_flow(kernel) {
        Ok(()) => {}
        Err(error @ PcuTypedDispatchValidationError::UnsupportedOperation(position)) => {
            if matches!(
                kernel.ops.get(position),
                Some(PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    index: PcuDispatchIndex::GridStrideId,
                    ..
                }))
            ) {
                return Err(PcuCpuHostError::InvalidValueFlow(error));
            }
            return Err(PcuCpuHostError::UnsupportedProfile);
        }
        Err(error) => return Err(PcuCpuHostError::InvalidValueFlow(error)),
    }
    let body = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { body, .. },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => *body,
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => body,
        _ => return Err(PcuCpuHostError::MissingReturn),
    };
    for op in body {
        let (reference, required) = match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, .. }) => {
                (*binding, PcuBindingAccess::ReadOnly)
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, .. }) => {
                (*binding, PcuBindingAccess::WriteOnly)
            }
            _ => continue,
        };
        if let Some(binding) = kernel
            .bindings
            .iter()
            .find(|binding| binding.reference() == reference)
            && matches!(
                (required, binding.access),
                (PcuBindingAccess::ReadOnly, PcuBindingAccess::WriteOnly)
                    | (PcuBindingAccess::WriteOnly, PcuBindingAccess::ReadOnly)
            )
        {
            return Err(PcuCpuHostError::InvalidBindingAccess {
                binding: reference,
                access: binding.access,
                required,
            });
        }
    }
    if kernel.entry.logical_shape[1..] != [1, 1] {
        return Err(PcuCpuHostError::UnsupportedProfile);
    }
    Ok(body)
}

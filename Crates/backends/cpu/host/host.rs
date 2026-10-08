//! Static cold selection of the admitted CPU host profiles.

#[rustfmt::skip]
use fusion_pcu::{
    validate_typed_dispatch_value_flow,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
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
    PcuScalarIdentityValidationError,
    CheckedFloatConversionMapValidationError,
    PcuTypedDispatchValidationError,
    PcuValueType,
    PcuRangePolicy,
};
#[rustfmt::skip]
use crate::{
    PcuCpuCheckedUnary,
    PcuCpuPreparedUnary,
    PcuCpuPreparedDivRem,
    PcuCpuIdentity,
    PcuCpuPreparedIdentity,
    PcuCpuCheckedConversion,
    PcuCpuPreparedConversion,
    PcuCpuCheckedBinary,
    PcuCpuPreparedBinary,
    PcuCpuPreparedBinaryError,
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
    HeaderUnderflowMismatch,
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
    Binary(PcuCpuPreparedBinaryError),
    Identity(PcuScalarIdentityValidationError),
    Conversion(CheckedFloatConversionMapValidationError),
    Fault(PcuExecutionFault),
    Integer(PcuCpuCheckedIntegerError),
    #[cfg(any(feature = "std", feature = "tensor"))]
    Composed(crate::PcuCpuComposedMapError),
    #[cfg(any(feature = "std", feature = "tensor"))]
    Transport(crate::PcuCpuScalarTransportError),
}

impl PcuCpuHostError {
    #[must_use]
    pub const fn fault(self) -> Option<PcuExecutionFault> {
        match self {
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Composed(error) => error.fault(),
            Self::Neg(PcuCpuPreparedNegError::Fault(fault))
            | Self::Integer(PcuCpuCheckedIntegerError::Fault(fault))
            | Self::Binary(PcuCpuPreparedBinaryError::Fault(fault))
            | Self::Fault(fault) => Some(fault),
            _ => None,
        }
    }
}

/// Explicit CPU host backend; instruction choice is resolved only during construction/preparation.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuHostBackend {
    processor: PcuCpuProcessor,
    neg: PcuCpuCheckedNeg,
}

impl PcuCpuHostBackend {
    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            processor: PcuCpuProcessor::scalar(),
            neg: PcuCpuCheckedNeg::scalar(),
        }
    }

    /// Constructs a backend with an explicit, already detected Neg instruction implementation.
    ///
    /// # Errors
    /// Rejects unavailable Neg instructions. Native-width Add/Sub choose proved SSE2/NEON cold profiles.
    pub const fn new(
        processor: PcuCpuProcessor,
        implementation: PcuCpuImplementation,
    ) -> Result<Self, PcuCpuImplementationUnavailable> {
        match PcuCpuCheckedNeg::new(processor, implementation) {
            Ok(neg) => Ok(Self { processor, neg }),
            Err(error) => Err(error),
        }
    }

    /// Detects available instructions outside warm calls; Mul/wide retain independent scalar profiles.
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
#[derive(Debug, Clone)]
pub enum PcuCpuPreparedHost {
    #[cfg(any(feature = "std", feature = "tensor"))]
    Transport(crate::PcuCpuPreparedScalarTransport),
    #[cfg(any(feature = "std", feature = "tensor"))]
    Composed(crate::PcuCpuPreparedComposedMap),
    Unary(PcuCpuPreparedUnary),
    #[cfg(any(feature = "std", feature = "tensor"))]
    UnaryRoles(crate::PcuCpuPreparedUnaryRoles),
    DivRem(PcuCpuPreparedDivRem),
    Neg(PcuCpuPreparedNeg),
    Identity(PcuCpuPreparedIdentity),
    Conversion(PcuCpuPreparedConversion),
    F32Binary(PcuCpuPreparedBinary<f32>),
    F64Binary(PcuCpuPreparedBinary<f64>),
    F16Binary(PcuCpuPreparedBinary<PcuF16Bits>),
    BF16Binary(PcuCpuPreparedBinary<PcuBf16Bits>),
    F8E4M3FNBinary(PcuCpuPreparedBinary<PcuF8E4M3FnBits>),
    F8E5M2Binary(PcuCpuPreparedBinary<PcuF8E5M2Bits>),
    I8(PcuCpuPreparedInteger<i8>),
    U8(PcuCpuPreparedInteger<u8>),
    I16(PcuCpuPreparedInteger<i16>),
    U16(PcuCpuPreparedInteger<u16>),
    I32(PcuCpuPreparedInteger<i32>),
    U32(PcuCpuPreparedInteger<u32>),
    I64(PcuCpuPreparedInteger<i64>),
    U64(PcuCpuPreparedInteger<u64>),
    I128(PcuCpuPreparedInteger<i128>),
    U128(PcuCpuPreparedInteger<u128>),
    I256(PcuCpuPreparedInteger<PcuI256>),
    U256(PcuCpuPreparedInteger<PcuU256>),
    I512(PcuCpuPreparedInteger<PcuI512>),
    U512(PcuCpuPreparedInteger<PcuU512>),
}

impl PcuCpuPreparedHost {
    /// Range publication contract captured by this exact scalar executable.
    #[must_use]
    pub const fn range_policy(&self) -> PcuRangePolicy {
        match self {
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Composed(p) => p.range_policy(),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Transport(p) => p.range_policy(),
            Self::Unary(p) => p.range_policy(),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::UnaryRoles(p) => p.range_policy(),
            Self::Conversion(p) => p.range_policy(),
            Self::F32Binary(p) => p.range_policy(),
            Self::F64Binary(p) => p.range_policy(),
            Self::F16Binary(p) => p.range_policy(),
            Self::BF16Binary(p) => p.range_policy(),
            Self::F8E4M3FNBinary(p) => p.range_policy(),
            Self::F8E5M2Binary(p) => p.range_policy(),
            Self::I8(p) => p.range_policy(),
            Self::U8(p) => p.range_policy(),
            Self::I16(p) => p.range_policy(),
            Self::U16(p) => p.range_policy(),
            Self::I32(p) => p.range_policy(),
            Self::U32(p) => p.range_policy(),
            Self::I64(p) => p.range_policy(),
            Self::U64(p) => p.range_policy(),
            Self::I128(p) => p.range_policy(),
            Self::U128(p) => p.range_policy(),
            Self::I256(p) => p.range_policy(),
            Self::U256(p) => p.range_policy(),
            Self::I512(p) => p.range_policy(),
            Self::U512(p) => p.range_policy(),
            Self::DivRem(_) | Self::Neg(_) | Self::Identity(_) => PcuRangePolicy::Reject,
        }
    }

    /// Number of complete arguments in this frozen prepared host schema.
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        match self {
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Composed(p) => p.argument_count(),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Transport(p) => p.argument_count(),
            Self::DivRem(p) => p.argument_count(),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::UnaryRoles(p) => p.argument_count(),
            Self::Unary(_) | Self::Neg(_) | Self::Identity(_) | Self::Conversion(_) => 2,
            Self::F32Binary(p) => p.argument_count(),
            Self::F64Binary(p) => p.argument_count(),
            Self::F16Binary(p) => p.argument_count(),
            Self::BF16Binary(p) => p.argument_count(),
            Self::F8E4M3FNBinary(p) => p.argument_count(),
            Self::F8E5M2Binary(p) => p.argument_count(),
            Self::I8(p) => p.argument_count(),
            Self::U8(p) => p.argument_count(),
            Self::I16(p) => p.argument_count(),
            Self::U16(p) => p.argument_count(),
            Self::I32(p) => p.argument_count(),
            Self::U32(p) => p.argument_count(),
            Self::I64(p) => p.argument_count(),
            Self::U64(p) => p.argument_count(),
            Self::I128(p) => p.argument_count(),
            Self::U128(p) => p.argument_count(),
            Self::I256(p) => p.argument_count(),
            Self::U256(p) => p.argument_count(),
            Self::I512(p) => p.argument_count(),
            Self::U512(p) => p.argument_count(),
        }
    }
}

impl PcuHostKernelBackend for PcuCpuHostBackend {
    type Prepared = PcuCpuPreparedHost;
    type Error = PcuCpuHostError;

    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        // Reject once before primitive/composed/transport fallback can inspect IR.
        if kernel.has_nested_grid_stride_loop() {
            return Err(PcuCpuHostError::UnsupportedProfile);
        }
        let primitive = self.prepare_primitive(kernel);
        #[cfg(any(feature = "std", feature = "tensor"))]
        if matches!(
            primitive,
            Err(PcuCpuHostError::UnsupportedProfile
                | PcuCpuHostError::Neg(PcuCpuPreparedNegError::UnsupportedProfile)
                | PcuCpuHostError::Binary(PcuCpuPreparedBinaryError::UnsupportedProfile)
                | PcuCpuHostError::Integer(
                    PcuCpuCheckedIntegerError::UnsupportedProfile
                        | PcuCpuCheckedIntegerError::InvalidKernel(
                            fusion_pcu::IntegerMapValidationError::InvalidValue(_)
                        )
                ))
        ) {
            // Structural admission, not arithmetic count, decides the primitive profile.
            // A detached plan keeps even unused checked results observable. Its cold
            // failure must not replace the original typed primitive refusal.
            // Integer primitive maps require storing their arithmetic result; a valid
            // saved SSA store instead reports InvalidValue in that narrow descriptor.
            // Only complete composed resource/type/SSA admission can replace it below.
            if let Ok(plan) = crate::composed::prepare_erased(kernel) {
                return Ok(PcuCpuPreparedHost::Composed(plan));
            }
            // A separate representation-only contract never bypasses checked arithmetic.
            if let Ok(plan) = crate::PcuCpuPreparedScalarTransport::prepare(kernel) {
                return Ok(PcuCpuPreparedHost::Transport(plan));
            }
        }
        primitive
    }
}

impl PcuCpuHostBackend {
    #[allow(clippy::too_many_lines)] // One closed scalar type selection retains explicit provider gates for every static variant.
    fn prepare_primitive(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<PcuCpuPreparedHost, PcuCpuHostError> {
        let body = validated_region(kernel)?;
        if let Some(value_type) = body.iter().find_map(|op| match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { value_type, .. }) => {
                Some(*value_type)
            }
            _ => None,
        }) {
            return PcuCpuPreparedDivRem::prepare(kernel, value_type)
                .map(PcuCpuPreparedHost::DivRem);
        }
        if matches!(
            body,
            [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. })
            ]
        ) {
            return PcuCpuIdentity
                .prepare_host_kernel(kernel)
                .map(PcuCpuPreparedHost::Identity);
        }
        if body.iter().any(|op| {
            matches!(
                op,
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert { .. })
            )
        }) {
            return PcuCpuCheckedConversion
                .prepare_host_kernel(kernel)
                .map(PcuCpuPreparedHost::Conversion);
        }
        if let Some(PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type,
            op,
            range_policy,
            ..
        })) = body.get(1)
        {
            macro_rules! unary {
                ($ty:ty) => {{
                    #[cfg(any(feature = "std", feature = "tensor"))]
                    if kernel.bindings.len() > 2 {
                        return crate::PcuCpuPreparedUnaryRoles::prepare::<$ty>(kernel)
                            .map(PcuCpuPreparedHost::UnaryRoles);
                    }
                    PcuCpuCheckedUnary::<$ty>::new()
                        .prepare_host_kernel(kernel)
                        .map(PcuCpuPreparedHost::Unary)
                }};
            }
            let scalar_unary = kernel
                .numerical_requirements
                .numerical_options
                .reproducibility
                == fusion_pcu::PcuReproducibility::PortableV1
                || kernel.bindings.len() > 2
                || *op == PcuDispatchFloatUnaryOp::Relu
                || *range_policy == PcuRangePolicy::Clamp
                || matches!(
                    body.first(),
                    Some(PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                        index: PcuDispatchIndex::BindingElementZero,
                        ..
                    }))
                );
            match value_type {
                PcuValueType::Scalar(PcuScalarType::F32) if scalar_unary => return unary!(f32),
                PcuValueType::Scalar(PcuScalarType::F64) if scalar_unary => return unary!(f64),
                PcuValueType::Scalar(PcuScalarType::F16) => return unary!(PcuF16Bits),
                PcuValueType::Scalar(PcuScalarType::BF16) => return unary!(PcuBf16Bits),
                PcuValueType::Scalar(PcuScalarType::F8E4M3FN) => return unary!(PcuF8E4M3FnBits),
                PcuValueType::Scalar(PcuScalarType::F8E5M2) => return unary!(PcuF8E5M2Bits),
                _ => {}
            }
        }
        let mut operations = body.iter().filter_map(|op| match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                value_type,
                op: PcuDispatchFloatUnaryOp::Neg,
                ..
            }) => Some((*value_type, true)),
            PcuDispatchOp::Data(
                PcuDispatchDataOp::CheckedFloatBinary { value_type, .. }
                | PcuDispatchDataOp::CheckedIntegerBinary { value_type, .. },
            ) => Some((*value_type, false)),
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
                PcuCpuCheckedInteger::<$ty>::for_processor(self.processor)
                    .prepare_host_kernel(kernel)
                    .map(PcuCpuPreparedHost::$variant)
                    .map_err(PcuCpuHostError::Integer)
            };
        }
        macro_rules! binary {
            ($ty:ty, $variant:ident) => {
                PcuCpuCheckedBinary::<$ty>::new()
                    .prepare_host_kernel(kernel)
                    .map(PcuCpuPreparedHost::$variant)
                    .map_err(PcuCpuHostError::Binary)
            };
        }
        match value_type {
            PcuValueType::Scalar(PcuScalarType::F32) => binary!(f32, F32Binary),
            PcuValueType::Scalar(PcuScalarType::F64) => binary!(f64, F64Binary),
            PcuValueType::Scalar(PcuScalarType::F16) => binary!(PcuF16Bits, F16Binary),
            PcuValueType::Scalar(PcuScalarType::BF16) => binary!(PcuBf16Bits, BF16Binary),
            PcuValueType::Scalar(PcuScalarType::F8E4M3FN) => {
                binary!(PcuF8E4M3FnBits, F8E4M3FNBinary)
            }
            PcuValueType::Scalar(PcuScalarType::F8E5M2) => {
                binary!(PcuF8E5M2Bits, F8E5M2Binary)
            }
            PcuValueType::Scalar(PcuScalarType::I8) => integer!(i8, I8),
            PcuValueType::Scalar(PcuScalarType::U8) => integer!(u8, U8),
            PcuValueType::Scalar(PcuScalarType::I16) => integer!(i16, I16),
            PcuValueType::Scalar(PcuScalarType::U16) => integer!(u16, U16),
            PcuValueType::Scalar(PcuScalarType::I32) => integer!(i32, I32),
            PcuValueType::Scalar(PcuScalarType::U32) => integer!(u32, U32),
            PcuValueType::Scalar(PcuScalarType::I64) => integer!(i64, I64),
            PcuValueType::Scalar(PcuScalarType::U64) => integer!(u64, U64),
            PcuValueType::Scalar(PcuScalarType::I128) => integer!(i128, I128),
            PcuValueType::Scalar(PcuScalarType::U128) => integer!(u128, U128),
            PcuValueType::Scalar(PcuScalarType::I256) => integer!(PcuI256, I256),
            PcuValueType::Scalar(PcuScalarType::U256) => integer!(PcuU256, U256),
            PcuValueType::Scalar(PcuScalarType::I512) => integer!(PcuI512, I512),
            PcuValueType::Scalar(PcuScalarType::U512) => integer!(PcuU512, U512),
            _ => Err(PcuCpuHostError::UnsupportedProfile),
        }
    }
}

type HostSchema = [(PcuBindingRef, usize, PcuBindingAccess); 3];
const fn binary_schema<T: fusion_pcu::PcuCheckedFloat>(
    prepared: &PcuCpuPreparedBinary<T>,
) -> (HostSchema, usize, PcuScalarType, usize) {
    (
        prepared.host_schema(),
        prepared.argument_count(),
        T::TYPE,
        T::HOST_SIZE,
    )
}

const fn integer_schema<T: fusion_pcu::PcuCheckedInteger>(
    prepared: &crate::PcuCpuPreparedInteger<T>,
) -> (HostSchema, usize, PcuScalarType, usize) {
    (
        prepared.host_schema(),
        prepared.argument_count(),
        T::TYPE,
        T::HOST_SIZE,
    )
}

impl PcuPreparedHostKernel for PcuCpuPreparedHost {
    type Error = PcuCpuHostError;

    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        if let Some(result) = self.call_detached(arguments) {
            return result;
        }
        let (schema, count, scalar, size) = match self {
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Composed(_) => unreachable!("detached composed schema already handled"),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Transport(_) => unreachable!("detached transport schema already handled"),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::UnaryRoles(_) => unreachable!("owned unary schema already handled"),
            Self::Unary(_) => unreachable!("frozen unary schema already handled"),
            Self::DivRem(_) => unreachable!("four-argument schema already handled"),
            Self::Conversion(_) => unreachable!("mixed schema validated by frozen conversion"),
            Self::Neg(prepared) => (
                prepared.host_schema(),
                2,
                prepared.scalar_type(),
                prepared.element_size(),
            ),
            Self::Identity(prepared) => (
                prepared.host_schema(),
                2,
                prepared.scalar_type(),
                prepared.element_size(),
            ),
            Self::F32Binary(prepared) => binary_schema(prepared),
            Self::F64Binary(prepared) => binary_schema(prepared),
            Self::F16Binary(prepared) => binary_schema(prepared),
            Self::BF16Binary(prepared) => binary_schema(prepared),
            Self::F8E4M3FNBinary(prepared) => binary_schema(prepared),
            Self::F8E5M2Binary(prepared) => binary_schema(prepared),
            Self::I8(prepared) => integer_schema(prepared),
            Self::U8(prepared) => integer_schema(prepared),
            Self::I16(prepared) => integer_schema(prepared),
            Self::U16(prepared) => integer_schema(prepared),
            Self::I32(prepared) => integer_schema(prepared),
            Self::U32(prepared) => integer_schema(prepared),
            Self::I64(prepared) => integer_schema(prepared),
            Self::U64(prepared) => integer_schema(prepared),
            Self::I128(prepared) => integer_schema(prepared),
            Self::U128(prepared) => integer_schema(prepared),
            Self::I256(prepared) => integer_schema(prepared),
            Self::U256(prepared) => integer_schema(prepared),
            Self::I512(prepared) => integer_schema(prepared),
            Self::U512(prepared) => integer_schema(prepared),
        };
        validate_arguments(arguments, &schema[..count], scalar, size)
            .map_err(PcuCpuHostError::Arguments)?;
        macro_rules! integer {
            ($prepared:ident) => {
                $prepared.call(arguments).map_err(PcuCpuHostError::Integer)
            };
        }
        match self {
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Composed(_) => unreachable!("detached composed call already completed"),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Transport(_) => unreachable!("detached transport call already completed"),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::UnaryRoles(_) => unreachable!("owned unary call already completed"),
            Self::Unary(_) => unreachable!("frozen unary call already completed"),
            Self::DivRem(_) => unreachable!("DivRem already completed"),
            Self::Identity(prepared) => prepared.call(arguments),
            Self::Conversion(_) => unreachable!("conversion already completed"),
            Self::Neg(prepared) => prepared.call(arguments).map_err(PcuCpuHostError::Neg),
            Self::F32Binary(prepared) => prepared.call(arguments).map_err(PcuCpuHostError::Binary),
            Self::F64Binary(prepared) => prepared.call(arguments).map_err(PcuCpuHostError::Binary),
            Self::F16Binary(prepared) => prepared.call(arguments).map_err(PcuCpuHostError::Binary),
            Self::BF16Binary(prepared) => prepared.call(arguments).map_err(PcuCpuHostError::Binary),
            Self::F8E4M3FNBinary(prepared) => {
                prepared.call(arguments).map_err(PcuCpuHostError::Binary)
            }
            Self::F8E5M2Binary(prepared) => {
                prepared.call(arguments).map_err(PcuCpuHostError::Binary)
            }
            Self::I8(prepared) => integer!(prepared),
            Self::U8(prepared) => integer!(prepared),
            Self::I16(prepared) => integer!(prepared),
            Self::U16(prepared) => integer!(prepared),
            Self::I32(prepared) => integer!(prepared),
            Self::U32(prepared) => integer!(prepared),
            Self::I64(prepared) => integer!(prepared),
            Self::U64(prepared) => integer!(prepared),
            Self::I128(prepared) => integer!(prepared),
            Self::U128(prepared) => integer!(prepared),
            Self::I256(prepared) => integer!(prepared),
            Self::U256(prepared) => integer!(prepared),
            Self::I512(prepared) => integer!(prepared),
            Self::U512(prepared) => integer!(prepared),
        }
    }
}

pub fn validate_arguments(
    arguments: &[PcuHostArgument<'_>],
    schema: &[(PcuBindingRef, usize, PcuBindingAccess)],
    scalar: PcuScalarType,
    size: usize,
) -> Result<(), PcuCpuHostArgumentError> {
    validate_arguments_with_indices(arguments, schema, scalar, size, |_, _| {})
}

// Resolve declaration positions while checking the existing argument contract. The
// callback observes only declarations that passed all checks; positions are call-local.
pub fn validate_arguments_with_indices(
    arguments: &[PcuHostArgument<'_>],
    schema: &[(PcuBindingRef, usize, PcuBindingAccess)],
    scalar: PcuScalarType,
    size: usize,
    mut resolved: impl FnMut(usize, usize),
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
    for (declaration, &(binding, elements, expected_access)) in schema.iter().enumerate() {
        let (position, argument) = arguments
            .iter()
            .enumerate()
            .find(|(_, argument)| argument.target() == binding)
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
        resolved(declaration, position);
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

pub fn validated_region<'a>(
    kernel: &PcuDispatchKernelIr<'a>,
) -> Result<&'a [PcuDispatchOp<'a>], PcuCpuHostError> {
    // Refuse nesting before typed/capability scans can follow cyclic public IR.
    let bounded_body = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { body, .. },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => *body,
        _ => kernel.ops,
    };
    if bounded_body
        .iter()
        .any(|operation| matches!(operation, PcuDispatchOp::GridStrideLoop { .. }))
    {
        return Err(PcuCpuHostError::UnsupportedProfile);
    }
    validate_portable_region(kernel)?;
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

fn validate_portable_region(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), PcuCpuHostError> {
    // Cold structural eligibility is necessary; typed preparation supplies the proved executor.
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == fusion_pcu::PcuReproducibility::PortableV1
    {
        #[cfg(any(feature = "std", feature = "tensor"))]
        if fusion_pcu::describe_portable_v1_checked_integer_composed_map::<4>(kernel).is_ok() {
            return Ok(());
        }
        if fusion_pcu::describe_portable_v1_integer_map(kernel).is_ok()
            || fusion_pcu::describe_portable_v1_integer_div_rem_map(kernel).is_ok()
            || fusion_pcu::describe_portable_v1_unary_map(kernel).is_ok()
        {
            return Ok(());
        }
        let description =
            fusion_pcu::describe_portable_v1_map(kernel).map_err(|error| match error {
                fusion_pcu::PcuPortableV1MapError::InvalidValueFlow(error) => {
                    PcuCpuHostError::InvalidValueFlow(error)
                }
                fusion_pcu::PcuPortableV1MapError::InvalidLogicalShape(shape) => {
                    PcuCpuHostError::InvalidLogicalShape(shape)
                }
                _ => PcuCpuHostError::UnsupportedProfile,
            })?;
        if !matches!(
            description.scalar,
            PcuScalarType::F16
                | PcuScalarType::BF16
                | PcuScalarType::F8E4M3FN
                | PcuScalarType::F8E5M2
        ) {
            return Err(PcuCpuHostError::UnsupportedProfile);
        }
    }
    Ok(())
}

impl PcuCpuPreparedHost {
    #[inline]
    fn call_detached(
        &mut self,
        arguments: &mut [PcuHostArgument<'_>],
    ) -> Option<Result<(), PcuCpuHostError>> {
        Some(match self {
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Transport(plan) => plan.call(arguments).map_err(PcuCpuHostError::Transport),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::Composed(plan) => plan.call(arguments).map_err(PcuCpuHostError::Composed),
            #[cfg(any(feature = "std", feature = "tensor"))]
            Self::UnaryRoles(plan) => plan.call(arguments),
            Self::DivRem(plan) => plan.call(arguments),
            Self::Conversion(plan) => plan.call(arguments),
            Self::Unary(plan) => plan.call(arguments),
            _ => return None,
        })
    }
}

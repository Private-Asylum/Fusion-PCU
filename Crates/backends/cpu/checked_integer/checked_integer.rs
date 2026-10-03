//! Typed, transactional scalar and cold-selected native SIMD checked integer maps.

use core::marker::PhantomData;
#[rustfmt::skip]
use crate::{
    PcuCpuProcessor,
    PcuCpuImplementation,
    PcuCpuImplementationUnavailable,
};
#[path = "execution/execution.rs"]
mod execution;
#[path = "offers/offers.rs"]
mod offers;
#[rustfmt::skip]
pub use offers::{
    PcuCpuIntegerOfferError,
    PcuCpuIntegerOffers,
};
#[rustfmt::skip]
use fusion_pcu::{
    validate_dispatch_submission,
    validate_integer_checked_binary_kernel,
    assess_checked_integer_binary_operands,
    CheckedIntegerBinaryOperandSchema,
    validate_typed_dispatch_value_flow,
    IntegerMapValidationError,
    PcuBindingAccess,
    PcuBindingRef,
    PcuCheckedInteger,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuSynchronousHostDispatchBackend,
    PcuTypedDispatchValidationError,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Revision of the cold-frozen checked integer scalar operation/layout executor.
pub const PCU_CPU_INTEGER_IMPLEMENTATION_REVISION: u64 = 2;

/// Admission, complete host-schema, or first terminal arithmetic failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuCheckedIntegerError {
    InvalidSubmission,
    InvalidLogicalShape([u32; 3]),
    InvalidGridExtent,
    InvalidKernel(IntegerMapValidationError),
    InvalidValueFlow(PcuTypedDispatchValidationError),
    UnsupportedProfile,
    ImplementationUnavailable(PcuCpuImplementationUnavailable),
    InvalidArguments,
    /// Lowest logical invocation; Reject preserves output, completed Clamp publishes useful output.
    Fault(PcuExecutionFault),
}

/// Typed synchronous reference errors share the exact prepared execution contract.
pub type PcuCheckedIntegerReferenceError = PcuCpuCheckedIntegerError;

/// Explicit CPU backend for fourteen checked integer representations and Add/Sub/Mul.
///
/// `new` is an independent scalar control. Optional native-width Add/Sub SIMD is frozen cold;
/// unavailable explicit instructions reject without substitution. Preparation captures a complete
/// canonical schema and actual SSA operand mapping, independently of the temporary IR.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuCheckedInteger<T: PcuCheckedInteger> {
    selection: Selection,
    marker: PhantomData<T>,
}

#[derive(Debug, Clone, Copy)]
enum Selection {
    Scalar,
    Explicit(PcuCpuImplementation),
    Available(PcuCpuProcessor),
}

impl<T: PcuCheckedInteger> PcuCpuCheckedInteger<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            selection: Selection::Scalar,
            marker: PhantomData,
        }
    }
    /// Admits an explicit bounded Add/Sub instruction implementation during cold preparation.
    /// # Errors
    /// Rejects unavailable ISA or unsupported AVX2; later preparation rejects wide/Mul profiles.
    pub const fn with_implementation(
        processor: PcuCpuProcessor,
        implementation: PcuCpuImplementation,
    ) -> Result<Self, PcuCpuCheckedIntegerError> {
        if matches!(implementation, PcuCpuImplementation::Avx2) {
            return Err(PcuCpuCheckedIntegerError::UnsupportedProfile);
        }
        match processor.require(implementation) {
            Ok(()) => Ok(Self {
                selection: Selection::Explicit(implementation),
                marker: PhantomData,
            }),
            Err(error) => Err(PcuCpuCheckedIntegerError::ImplementationUnavailable(error)),
        }
    }
    pub(crate) const fn for_processor(processor: PcuCpuProcessor) -> Self {
        Self {
            selection: Selection::Available(processor),
            marker: PhantomData,
        }
    }
    fn select(
        self,
        op: PcuDispatchIntegerBinaryOp,
    ) -> Result<PcuCpuImplementation, PcuCpuCheckedIntegerError> {
        let native = T::HOST_SIZE <= 8
            && matches!(
                op,
                PcuDispatchIntegerBinaryOp::Add | PcuDispatchIntegerBinaryOp::Sub
            )
            && cfg!(target_endian = "little");
        match self.selection {
            Selection::Scalar => Ok(PcuCpuImplementation::Scalar),
            Selection::Explicit(implementation) => {
                if native || implementation == PcuCpuImplementation::Scalar {
                    Ok(implementation)
                } else {
                    Err(PcuCpuCheckedIntegerError::UnsupportedProfile)
                }
            }
            Selection::Available(processor) => {
                #[cfg(not(any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "aarch64"
                )))]
                let _ = processor;
                if native {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if processor.features().sse2 {
                        return Ok(PcuCpuImplementation::Sse2);
                    }
                    #[cfg(target_arch = "aarch64")]
                    if processor.features().neon {
                        return Ok(PcuCpuImplementation::Neon);
                    }
                }
                Ok(PcuCpuImplementation::Scalar)
            }
        }
    }
}

impl<T: PcuCheckedInteger> Default for PcuCpuCheckedInteger<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Caller-owned typed-slice reference adapter with the same transactional checked semantics.
#[derive(Debug, Clone, Copy)]
pub struct PcuCheckedIntegerReference<T: PcuCheckedInteger> {
    marker: PhantomData<T>,
}

impl<T: PcuCheckedInteger> PcuCheckedIntegerReference<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T: PcuCheckedInteger> Default for PcuCheckedIntegerReference<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct Input {
    binding: PcuBindingRef,
}

/// Detached scalar/SIMD executable; no allocation, IR lifetime, or host pointer is retained.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuPreparedInteger<T: PcuCheckedInteger> {
    inputs: [Input; 2],
    schema: [(PcuBindingRef, usize, PcuBindingAccess); 3],
    argument_count: usize,
    output: PcuBindingRef,
    operands: [usize; 2],
    op: PcuDispatchIntegerBinaryOp,
    local_id: u32,
    range: PcuRangePolicy,
    extent: usize,
    execute: execution::Executable,
    implementation: PcuCpuImplementation,
    marker: PhantomData<T>,
}

impl<T: PcuCheckedInteger> PcuCpuPreparedInteger<T> {
    pub(super) const fn host_schema(&self) -> [(PcuBindingRef, usize, PcuBindingAccess); 3] {
        self.schema
    }
    /// Number of original typed declarations; schema-proved unused reads retain zero spans.
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.argument_count
    }
    #[must_use]
    pub const fn operation(&self) -> PcuDispatchIntegerBinaryOp {
        self.op
    }

    /// Exact cold implementation ID; wide representations never alias primitive IDs.
    #[must_use]
    pub const fn local_id(&self) -> u32 {
        self.local_id
    }

    /// Frozen publication policy for the checked map.
    #[must_use]
    pub const fn range_policy(&self) -> PcuRangePolicy {
        self.range
    }

    /// Scalar Reject retains revision two; scalar Clamp and disjoint SIMD profiles use revision one.
    #[must_use]
    pub const fn implementation_revision(&self) -> u64 {
        if self.local_id >= 2048 || !matches!(self.implementation, PcuCpuImplementation::Scalar) {
            return 1;
        }
        match self.range {
            PcuRangePolicy::Reject => PCU_CPU_INTEGER_IMPLEMENTATION_REVISION,
            PcuRangePolicy::Clamp => 1,
        }
    }

    /// Exact frozen ISA; scalar controls never inherit automatic host selection.
    #[must_use]
    pub const fn implementation(&self) -> PcuCpuImplementation {
        self.implementation
    }
}

impl<T: PcuCheckedInteger> PcuHostKernelBackend for PcuCpuCheckedInteger<T> {
    type Prepared = PcuCpuPreparedInteger<T>;
    type Error = PcuCpuCheckedIntegerError;

    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let (body, extent) = validated_integer_region(kernel)?;
        // Closed representation admission also freezes a nonpanicking exact ID base.
        let base = reject_base(T::TYPE)?;
        let Some(PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            op,
            value_type,
            range_policy,
            ..
        })) = body.get(2)
        else {
            return Err(PcuCpuCheckedIntegerError::UnsupportedProfile);
        };
        if *value_type != PcuValueType::Scalar(T::TYPE) {
            return Err(PcuCpuCheckedIntegerError::UnsupportedProfile);
        }
        let roles = validate_integer_profile::<T>(kernel, *op)?;
        let extent =
            usize::try_from(extent).map_err(|_| PcuCpuCheckedIntegerError::UnsupportedProfile)?;
        let counts = roles.input_element_counts(extent);
        let loaded = roles.input_bindings();
        let inputs = [
            Input { binding: loaded[0] },
            Input {
                binding: *loaded.get(1).unwrap_or(&loaded[0]),
            },
        ];
        let mut schema = [(roles.output_binding(), 0, PcuBindingAccess::ReadOnly); 3];
        for (slot, binding) in kernel.bindings.iter().enumerate() {
            let target = binding.reference();
            schema[slot] = if target == roles.output_binding() {
                (target, extent, PcuBindingAccess::ReadWrite)
            } else {
                let count = loaded
                    .iter()
                    .position(|input| *input == target)
                    .map_or(0, |input| counts[input]);
                (target, count, PcuBindingAccess::ReadOnly)
            };
        }
        let legacy = validate_integer_checked_binary_kernel(
            kernel,
            *value_type,
            *op,
            PcuValueTypeCaps::for_scalar(T::TYPE),
        )
        .is_ok();
        let implementation = self.select(*op)?;
        let indices = roles.operand_indices();
        Ok(PcuCpuPreparedInteger {
            inputs,
            schema,
            argument_count: kernel.bindings.len(),
            output: roles.output_binding(),
            operands: roles.operand_inputs(),
            implementation,
            execute: execution::prepare_selected::<T>(
                implementation,
                *op,
                *range_policy,
                indices[0] == PcuDispatchIndex::BindingElementZero,
                indices[1] == PcuDispatchIndex::BindingElementZero,
            )?,
            op: *op,
            range: *range_policy,
            local_id: if kernel
                .numerical_requirements
                .numerical_options
                .reproducibility
                == fusion_pcu::PcuReproducibility::PortableV1
            {
                // Scalar/SSE2/NEON numerical mechanisms stay identical; cold profile IDs are separate.
                operand_id(T::TYPE, *op, *range_policy, implementation)? + 2048
            } else if legacy {
                implementation_id(T::TYPE, *op, *range_policy, implementation, base)?
            } else {
                operand_id(T::TYPE, *op, *range_policy, implementation)?
            },
            extent,
            marker: PhantomData,
        })
    }
}

impl<T: PcuCheckedInteger> PcuPreparedHostKernel for PcuCpuPreparedInteger<T> {
    type Error = PcuCpuCheckedIntegerError;

    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        let positions = self.validate_arguments(arguments)?;
        let destination = positions[2];
        let (before, destination_and_after) = arguments.split_at_mut(destination);
        let (output, after) = destination_and_after
            .split_first_mut()
            .expect("validated output position");
        let input = |position: usize| {
            if position < destination {
                before[position].bytes()
            } else {
                after[position - destination - 1].bytes()
            }
        };
        // Every original declaration was validated even if the SSA operands repeat one
        // input. The disjoint argument partitions retain safe Rust borrowing around the
        // single frozen typed executable; output bytes never alias an input span.
        (self.execute)(
            input(positions[self.operands[0]]),
            input(positions[self.operands[1]]),
            output.bytes_mut().expect("validated writable output"),
            self.extent,
        )
    }
}

impl<T: PcuCheckedInteger> PcuCpuPreparedInteger<T> {
    fn validate_arguments(
        &self,
        arguments: &[PcuHostArgument<'_>],
    ) -> Result<[usize; 3], PcuCpuCheckedIntegerError> {
        super::host::validate_arguments(
            arguments,
            &self.schema[..self.argument_count],
            T::TYPE,
            T::HOST_SIZE,
        )
        .map_err(|_| PcuCpuCheckedIntegerError::InvalidArguments)?;
        let refs = [self.inputs[0].binding, self.inputs[1].binding, self.output];
        let mut positions = [0; 3];
        for (slot, target) in refs.into_iter().enumerate() {
            positions[slot] = arguments
                .iter()
                .position(|argument| argument.target() == target)
                .ok_or(PcuCpuCheckedIntegerError::InvalidArguments)?;
        }
        Ok(positions)
    }
}

// SAFETY: Full submission/profile/schema precedes mutation. Reject preflights arithmetic;
// sealed Clamp Add/Sub/Mul always completes with an exact or defined useful range value.
// Execution is synchronous and never retains caller storage or a pointer after return.
unsafe impl<T: PcuCheckedInteger> PcuSynchronousHostDispatchBackend<T>
    for PcuCheckedIntegerReference<T>
{
    type Error = PcuCheckedIntegerReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, T>],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_dispatch_submission(submission)
            .map_err(|_| PcuCpuCheckedIntegerError::InvalidSubmission)?;
        if !parameters.is_empty() || !(2..=3).contains(&bindings.len()) {
            return Err(PcuCpuCheckedIntegerError::InvalidArguments);
        }
        let mut prepared =
            PcuCpuCheckedInteger::<T>::new().prepare_host_kernel(submission.kernel)?;
        match bindings {
            [first, second] => prepared.call(&mut [argument(first), argument(second)]),
            [first, second, third] => {
                prepared.call(&mut [argument(first), argument(second), argument(third)])
            }
            _ => Err(PcuCpuCheckedIntegerError::InvalidArguments),
        }
    }
}

const fn argument<'a, T: PcuCheckedInteger>(
    binding: &'a mut PcuHostScalarBinding<'_, T>,
) -> PcuHostArgument<'a> {
    match &mut binding.slice {
        PcuHostScalarSlice::Read(values) => PcuHostArgument::read(binding.target, values),
        PcuHostScalarSlice::ReadWrite(values) => {
            PcuHostArgument::read_write(binding.target, values)
        }
    }
}

pub fn fault(invocation: usize, kind: PcuExecutionFaultKind) -> PcuCpuCheckedIntegerError {
    PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
        recovered: false,
        kind,
        invocation_id: u64::try_from(invocation).expect("admitted u32 logical extent"),
    })
}

fn validate_integer_profile<T: PcuCheckedInteger>(
    kernel: &PcuDispatchKernelIr<'_>,
    op: PcuDispatchIntegerBinaryOp,
) -> Result<CheckedIntegerBinaryOperandSchema, PcuCpuCheckedIntegerError> {
    match assess_checked_integer_binary_operands(
        kernel,
        PcuValueType::Scalar(T::TYPE),
        op,
        PcuValueTypeCaps::for_scalar(T::TYPE),
    ) {
        Ok(schema) => Ok(schema),
        Err(
            IntegerMapValidationError::UnsupportedInterface
            | IntegerMapValidationError::UnsupportedRequirements
            | IntegerMapValidationError::UnsupportedOperation(_),
        ) => Err(PcuCpuCheckedIntegerError::UnsupportedProfile),
        Err(error) => Err(PcuCpuCheckedIntegerError::InvalidKernel(error)),
    }
}

pub fn validated_integer_region<'a>(
    kernel: &PcuDispatchKernelIr<'a>,
) -> Result<(&'a [PcuDispatchOp<'a>], u32), PcuCpuCheckedIntegerError> {
    // Admit only the independently proved exact fourteen-width map descriptor, never a general dialect.
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == fusion_pcu::PcuReproducibility::PortableV1
    {
        fusion_pcu::describe_portable_v1_integer_map(kernel)
            .map_err(|_| PcuCpuCheckedIntegerError::UnsupportedProfile)?;
    }
    if kernel.entry.logical_shape.contains(&0) {
        return Err(PcuCpuCheckedIntegerError::InvalidLogicalShape(
            kernel.entry.logical_shape,
        ));
    }
    if kernel.entry.logical_shape[1..] != [1, 1] {
        return Err(PcuCpuCheckedIntegerError::UnsupportedProfile);
    }
    let (body, extent) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (*body, *extent),
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (body, kernel.entry.logical_shape[0]),
        _ => {
            return Err(PcuCpuCheckedIntegerError::InvalidKernel(
                IntegerMapValidationError::MissingReturn,
            ));
        }
    };
    for (index, binding) in kernel.bindings.iter().enumerate() {
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == binding.reference())
        {
            return Err(PcuCpuCheckedIntegerError::InvalidKernel(
                IntegerMapValidationError::DuplicateBinding(binding.reference()),
            ));
        }
    }
    if extent == 0 {
        return Err(PcuCpuCheckedIntegerError::InvalidGridExtent);
    }
    validate_typed_dispatch_value_flow(kernel)
        .map_err(PcuCpuCheckedIntegerError::InvalidValueFlow)?;
    Ok((body, extent))
}

const fn clamp_base(scalar: fusion_pcu::PcuScalarType) -> Result<u32, PcuCpuCheckedIntegerError> {
    use fusion_pcu::PcuScalarType;
    Ok(match scalar {
        PcuScalarType::I8 => 320,
        PcuScalarType::U8 => 323,
        PcuScalarType::I16 => 326,
        PcuScalarType::U16 => 329,
        PcuScalarType::I32 => 332,
        PcuScalarType::U32 => 335,
        PcuScalarType::I64 => 338,
        PcuScalarType::U64 => 341,
        PcuScalarType::I128 => 344,
        PcuScalarType::U128 => 347,
        PcuScalarType::I256 => 350,
        PcuScalarType::U256 => 353,
        PcuScalarType::I512 => 356,
        PcuScalarType::U512 => 359,
        _ => return Err(PcuCpuCheckedIntegerError::UnsupportedProfile),
    })
}

const fn reject_base(scalar: fusion_pcu::PcuScalarType) -> Result<u32, PcuCpuCheckedIntegerError> {
    Ok(match scalar {
        fusion_pcu::PcuScalarType::I8 => 4,
        fusion_pcu::PcuScalarType::U8 => 7,
        fusion_pcu::PcuScalarType::I16 => 10,
        fusion_pcu::PcuScalarType::U16 => 13,
        fusion_pcu::PcuScalarType::I32 => 16,
        fusion_pcu::PcuScalarType::U32 => 19,
        fusion_pcu::PcuScalarType::I64 => 22,
        fusion_pcu::PcuScalarType::U64 => 25,
        fusion_pcu::PcuScalarType::I128 => 256,
        fusion_pcu::PcuScalarType::U128 => 259,
        fusion_pcu::PcuScalarType::I256 => 262,
        fusion_pcu::PcuScalarType::U256 => 265,
        fusion_pcu::PcuScalarType::I512 => 268,
        fusion_pcu::PcuScalarType::U512 => 271,
        _ => return Err(PcuCpuCheckedIntegerError::UnsupportedProfile),
    })
}

fn simd_id(
    scalar: fusion_pcu::PcuScalarType,
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    implementation: PcuCpuImplementation,
) -> Result<u32, PcuCpuCheckedIntegerError> {
    use fusion_pcu::PcuScalarType as S;
    let index = match scalar {
        S::I8 => 0,
        S::U8 => 1,
        S::I16 => 2,
        S::U16 => 3,
        S::I32 => 4,
        S::U32 => 5,
        S::I64 => 6,
        S::U64 => 7,
        _ => return Err(PcuCpuCheckedIntegerError::UnsupportedProfile),
    };
    let arch = match implementation {
        PcuCpuImplementation::Sse2 => 0,
        PcuCpuImplementation::Neon => 1,
        _ => return Err(PcuCpuCheckedIntegerError::UnsupportedProfile),
    };
    let operation = match op {
        PcuDispatchIntegerBinaryOp::Add => 0,
        PcuDispatchIntegerBinaryOp::Sub => 1,
        PcuDispatchIntegerBinaryOp::Mul => {
            return Err(PcuCpuCheckedIntegerError::UnsupportedProfile);
        }
    };
    Ok(1024
        + arch * 32
        + if range == PcuRangePolicy::Clamp {
            16
        } else {
            0
        }
        + index * 2
        + operation)
}

fn implementation_id(
    scalar: fusion_pcu::PcuScalarType,
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    implementation: PcuCpuImplementation,
    base: u32,
) -> Result<u32, PcuCpuCheckedIntegerError> {
    if implementation == PcuCpuImplementation::Scalar {
        let base = if range == PcuRangePolicy::Clamp {
            clamp_base(scalar)?
        } else {
            base
        };
        Ok(base
            + match op {
                PcuDispatchIntegerBinaryOp::Add => 0,
                PcuDispatchIntegerBinaryOp::Sub => 1,
                PcuDispatchIntegerBinaryOp::Mul => 2,
            })
    } else {
        simd_id(scalar, op, range, implementation)
    }
}

/// Disjoint opt-in IDs preserve all earlier distinct-input scalar and SIMD offers.
fn operand_id(
    scalar: fusion_pcu::PcuScalarType,
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    implementation: PcuCpuImplementation,
) -> Result<u32, PcuCpuCheckedIntegerError> {
    let reject = reject_base(scalar)?;
    let index = match scalar {
        fusion_pcu::PcuScalarType::I8 => 0,
        fusion_pcu::PcuScalarType::U8 => 1,
        fusion_pcu::PcuScalarType::I16 => 2,
        fusion_pcu::PcuScalarType::U16 => 3,
        fusion_pcu::PcuScalarType::I32 => 4,
        fusion_pcu::PcuScalarType::U32 => 5,
        fusion_pcu::PcuScalarType::I64 => 6,
        fusion_pcu::PcuScalarType::U64 => 7,
        _ => 8 + (reject - 256) / 3,
    };
    let operation = match op {
        PcuDispatchIntegerBinaryOp::Add => 0,
        PcuDispatchIntegerBinaryOp::Sub => 1,
        PcuDispatchIntegerBinaryOp::Mul => 2,
    };
    let clamp = u32::from(range == PcuRangePolicy::Clamp);
    match implementation {
        PcuCpuImplementation::Scalar => Ok(2048 + index * 6 + clamp * 3 + operation),
        PcuCpuImplementation::Sse2 | PcuCpuImplementation::Neon if index < 8 && operation < 2 => {
            Ok(if implementation == PcuCpuImplementation::Sse2 {
                2176
            } else {
                2304
            } + index * 4
                + clamp * 2
                + operation)
        }
        _ => Err(PcuCpuCheckedIntegerError::UnsupportedProfile),
    }
}

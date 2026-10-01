//! Typed, transactional scalar execution of canonical checked integer maps.

use core::marker::PhantomData;
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
    InvalidArguments,
    /// Lowest logical invocation; every output remains unchanged on failure.
    Fault(PcuExecutionFault),
}

/// Typed synchronous reference errors share the exact prepared execution contract.
pub type PcuCheckedIntegerReferenceError = PcuCpuCheckedIntegerError;

/// Explicit scalar CPU backend for all eight checked integer widths and Add/Sub/Mul.
///
/// No runtime instruction or backend fallback is selected. Preparation captures a complete
/// canonical schema and actual SSA operand mapping, independently of the temporary IR.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuCheckedInteger<T: PcuCheckedInteger> {
    marker: PhantomData<T>,
}

impl<T: PcuCheckedInteger> PcuCpuCheckedInteger<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
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
    broadcast: bool,
}

/// Detached scalar executable; no allocation, IR lifetime, or host pointer is retained.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuPreparedInteger<T: PcuCheckedInteger> {
    inputs: [Input; 2],
    output: PcuBindingRef,
    operands: [usize; 2],
    op: PcuDispatchIntegerBinaryOp,
    extent: usize,
    execute: execution::Executable,
    marker: PhantomData<T>,
}

impl<T: PcuCheckedInteger> PcuCpuPreparedInteger<T> {
    pub(super) const fn host_schema(&self) -> [(PcuBindingRef, usize, PcuBindingAccess); 3] {
        [
            (
                self.inputs[0].binding,
                self.required(0),
                PcuBindingAccess::ReadOnly,
            ),
            (
                self.inputs[1].binding,
                self.required(1),
                PcuBindingAccess::ReadOnly,
            ),
            (self.output, self.extent, PcuBindingAccess::ReadWrite),
        ]
    }
    #[must_use]
    pub const fn operation(&self) -> PcuDispatchIntegerBinaryOp {
        self.op
    }

    const fn required(&self, input: usize) -> usize {
        if self.inputs[input].broadcast {
            1
        } else {
            self.extent
        }
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
        let Some(PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            op,
            value_type,
            ..
        })) = body.get(2)
        else {
            return Err(PcuCpuCheckedIntegerError::UnsupportedProfile);
        };
        if *value_type != PcuValueType::Scalar(T::TYPE) {
            return Err(PcuCpuCheckedIntegerError::UnsupportedProfile);
        }
        validate_integer_profile::<T>(kernel, *op)?;
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: first,
                binding: first_binding,
                index: first_index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: second,
                binding: second_binding,
                index: second_index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { lhs, rhs, .. }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output, ..
            }),
        ] = body
        else {
            unreachable!("canonical validator admits exactly four operations")
        };
        let operand = |value| {
            if value == first {
                0
            } else {
                assert_eq!(
                    value, second,
                    "typed SSA permits only the two loaded inputs"
                );
                1
            }
        };
        Ok(PcuCpuPreparedInteger {
            inputs: [
                Input {
                    binding: *first_binding,
                    broadcast: *first_index == PcuDispatchIndex::BindingElementZero,
                },
                Input {
                    binding: *second_binding,
                    broadcast: *second_index == PcuDispatchIndex::BindingElementZero,
                },
            ],
            output: *output,
            operands: [operand(lhs), operand(rhs)],
            execute: execution::prepare::<T>(
                *op,
                if operand(lhs) == 0 {
                    *first_index == PcuDispatchIndex::BindingElementZero
                } else {
                    *second_index == PcuDispatchIndex::BindingElementZero
                },
                if operand(rhs) == 0 {
                    *first_index == PcuDispatchIndex::BindingElementZero
                } else {
                    *second_index == PcuDispatchIndex::BindingElementZero
                },
            ),
            op: *op,
            extent: usize::try_from(extent)
                .map_err(|_| PcuCpuCheckedIntegerError::UnsupportedProfile)?,
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
        // All three original schemas were validated even if the SSA operands repeat one
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
        if arguments.len() != 3 {
            return Err(PcuCpuCheckedIntegerError::InvalidArguments);
        }
        let refs = [self.inputs[0].binding, self.inputs[1].binding, self.output];
        let mut positions = [0; 3];
        for (slot, target) in refs.into_iter().enumerate() {
            let position = arguments
                .iter()
                .position(|argument| argument.target() == target)
                .ok_or(PcuCpuCheckedIntegerError::InvalidArguments)?;
            let argument = &arguments[position];
            let required = if slot == 2 {
                self.extent
            } else {
                self.required(slot)
            };
            let bytes = required
                .checked_mul(T::HOST_SIZE)
                .ok_or(PcuCpuCheckedIntegerError::InvalidArguments)?;
            let access = if slot == 2 {
                PcuBindingAccess::ReadWrite
            } else {
                PcuBindingAccess::ReadOnly
            };
            if argument.scalar() != T::TYPE
                || argument.access() != access
                || argument.bytes().len() < bytes
            {
                return Err(PcuCpuCheckedIntegerError::InvalidArguments);
            }
            positions[slot] = position;
        }
        Ok(positions)
    }
}

// SAFETY: Full submission/profile/schema and all arithmetic are checked before mutation.
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
        if !parameters.is_empty() || bindings.len() != 3 {
            return Err(PcuCpuCheckedIntegerError::InvalidArguments);
        }
        let mut prepared =
            PcuCpuCheckedInteger::<T>::new().prepare_host_kernel(submission.kernel)?;
        let [first, second, third] = bindings else {
            unreachable!("exact three-binding schema")
        };
        prepared.call(&mut [argument(first), argument(second), argument(third)])
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

fn fault(invocation: usize, kind: PcuExecutionFaultKind) -> PcuCpuCheckedIntegerError {
    PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
        recovered: false,
        kind,
        invocation_id: u64::try_from(invocation).expect("admitted u32 logical extent"),
    })
}

fn validate_integer_profile<T: PcuCheckedInteger>(
    kernel: &PcuDispatchKernelIr<'_>,
    op: PcuDispatchIntegerBinaryOp,
) -> Result<(), PcuCpuCheckedIntegerError> {
    match validate_integer_checked_binary_kernel(
        kernel,
        PcuValueType::Scalar(T::TYPE),
        op,
        PcuValueTypeCaps::for_scalar(T::TYPE),
    ) {
        Ok(()) => {}
        Err(
            IntegerMapValidationError::UnsupportedInterface
            | IntegerMapValidationError::UnsupportedRequirements
            | IntegerMapValidationError::UnsupportedOperation(_),
        ) => return Err(PcuCpuCheckedIntegerError::UnsupportedProfile),
        Err(error) => return Err(PcuCpuCheckedIntegerError::InvalidKernel(error)),
    }
    Ok(())
}

fn validated_integer_region<'a>(
    kernel: &PcuDispatchKernelIr<'a>,
) -> Result<(&'a [PcuDispatchOp<'a>], u32), PcuCpuCheckedIntegerError> {
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

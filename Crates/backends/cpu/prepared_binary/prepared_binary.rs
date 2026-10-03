//! Typed, transactional scalar execution of canonical checked float maps.

use core::marker::PhantomData;
#[path = "execution/execution.rs"]
mod execution;
#[rustfmt::skip]
use fusion_pcu::{
    validate_dispatch_submission,
    assess_checked_float_binary_operands,
    CheckedFloatBinaryOperandSchema,
    validate_typed_dispatch_value_flow,
    CheckedFloatBinaryMapValidationError,
    PcuBindingAccess,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchFloatBinaryOp,
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
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
};

/// Revision of the cold-frozen checked float scalar operation/layout executor.
pub const PCU_CPU_FLOAT_BINARY_IMPLEMENTATION_REVISION: u64 = 2;

/// Admission, complete host-schema, or first terminal arithmetic failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuPreparedBinaryError {
    InvalidSubmission,
    InvalidLogicalShape([u32; 3]),
    InvalidGridExtent,
    InvalidKernel(CheckedFloatBinaryMapValidationError),
    InvalidValueFlow(PcuTypedDispatchValidationError),
    UnsupportedProfile,
    InvalidArguments,
    /// First recovered range fault after complete publication, or first fatal fault before any store.
    /// Fatal faults preserve every output; recovered faults retain all useful clamped results.
    Fault(PcuExecutionFault),
}

/// Typed synchronous reference errors share the exact prepared execution contract.
pub type PcuCheckedBinaryReferenceError = PcuCpuPreparedBinaryError;

/// Explicit software CPU backend for six sealed checked Add/Sub/Mul/Div formats.
///
/// Binary16, BF16, E4M3FN and E5M2 use integer/rational destination-format arithmetic;
/// this admission does not claim hardware low-precision instructions or tensor support.
///
/// No runtime instruction or backend fallback is selected. Preparation captures a complete
/// canonical schema and actual SSA operand mapping, independently of the temporary IR.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuCheckedBinary<T: PcuCheckedFloat> {
    marker: PhantomData<T>,
}

impl<T: PcuCheckedFloat> PcuCpuCheckedBinary<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T: PcuCheckedFloat> Default for PcuCpuCheckedBinary<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Caller-owned typed-slice reference adapter with the same transactional checked semantics.
#[derive(Debug, Clone, Copy)]
pub struct PcuCheckedBinaryReference<T: PcuCheckedFloat> {
    marker: PhantomData<T>,
}

impl<T: PcuCheckedFloat> PcuCheckedBinaryReference<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T: PcuCheckedFloat> Default for PcuCheckedBinaryReference<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct Input {
    binding: PcuBindingRef,
}

/// Detached scalar executable; no allocation, IR lifetime, or host pointer is retained.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuPreparedBinary<T: PcuCheckedFloat> {
    inputs: [Input; 2],
    schema: [(PcuBindingRef, usize, PcuBindingAccess); 3],
    argument_count: usize,
    output: PcuBindingRef,
    operands: [usize; 2],
    op: PcuDispatchFloatBinaryOp,
    extent: usize,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    reproducibility: fusion_pcu::PcuReproducibility,
    execute: execution::Executable,
    marker: PhantomData<T>,
}

impl<T: PcuCheckedFloat> PcuCpuPreparedBinary<T> {
    pub(super) const fn host_schema(&self) -> [(PcuBindingRef, usize, PcuBindingAccess); 3] {
        self.schema
    }

    /// Number of declared typed arguments; unused read-only declarations retain zero spans.
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.argument_count
    }
    #[must_use]
    pub const fn operation(&self) -> PcuDispatchFloatBinaryOp {
        self.op
    }

    #[must_use]
    pub const fn range_policy(&self) -> PcuRangePolicy {
        self.range
    }

    /// Reproducibility admitted once by the cold executable constructor.
    #[must_use]
    pub const fn reproducibility(&self) -> fusion_pcu::PcuReproducibility {
        self.reproducibility
    }

    #[must_use]
    pub const fn underflow_policy(&self) -> PcuFloatUnderflowPolicy {
        self.underflow
    }
}

impl<T: PcuCheckedFloat> PcuHostKernelBackend for PcuCpuCheckedBinary<T> {
    type Prepared = PcuCpuPreparedBinary<T>;
    type Error = PcuCpuPreparedBinaryError;

    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let (body, extent) = validated_float_region(kernel)?;
        if !matches!(
            T::TYPE,
            fusion_pcu::PcuScalarType::F16
                | fusion_pcu::PcuScalarType::BF16
                | fusion_pcu::PcuScalarType::F32
                | fusion_pcu::PcuScalarType::F64
                | fusion_pcu::PcuScalarType::F8E4M3FN
                | fusion_pcu::PcuScalarType::F8E5M2
        ) {
            return Err(PcuCpuPreparedBinaryError::UnsupportedProfile);
        }
        let Some(PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            op,
            value_type,
            underflow_policy,
            range_policy,
            ..
        })) = body.get(2)
        else {
            return Err(PcuCpuPreparedBinaryError::UnsupportedProfile);
        };
        if *value_type != PcuValueType::Scalar(T::TYPE) {
            return Err(PcuCpuPreparedBinaryError::UnsupportedProfile);
        }
        let schema = validate_float_profile::<T>(kernel, *op, *underflow_policy)?;
        let extent =
            usize::try_from(extent).map_err(|_| PcuCpuPreparedBinaryError::UnsupportedProfile)?;
        let counts = schema.input_element_counts(extent);
        let loaded = schema.input_bindings();
        let inputs = [
            Input { binding: loaded[0] },
            Input {
                binding: *loaded.get(1).unwrap_or(&loaded[0]),
            },
        ];
        let mut declared = [(schema.output_binding(), 0, PcuBindingAccess::ReadOnly); 3];
        for (slot, binding) in kernel.bindings.iter().enumerate() {
            let target = binding.reference();
            declared[slot] = if target == schema.output_binding() {
                (target, extent, PcuBindingAccess::ReadWrite)
            } else {
                let count = loaded
                    .iter()
                    .position(|input| *input == target)
                    .map_or(0, |input| counts[input]);
                (target, count, PcuBindingAccess::ReadOnly)
            };
        }
        let indices = schema.operand_indices();
        Ok(PcuCpuPreparedBinary {
            inputs,
            schema: declared,
            argument_count: kernel.bindings.len(),
            output: schema.output_binding(),
            operands: schema.operand_inputs(),
            execute: execution::prepare::<T>(
                *op,
                *range_policy,
                indices[0] == PcuDispatchIndex::BindingElementZero,
                indices[1] == PcuDispatchIndex::BindingElementZero,
            ),
            op: *op,
            underflow: *underflow_policy,
            range: *range_policy,
            reproducibility: kernel
                .numerical_requirements
                .numerical_options
                .reproducibility,
            extent,
            marker: PhantomData,
        })
    }
}

impl<T: PcuCheckedFloat> PcuPreparedHostKernel for PcuCpuPreparedBinary<T> {
    type Error = PcuCpuPreparedBinaryError;

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
            self.underflow,
        )
    }
}

impl<T: PcuCheckedFloat> PcuCpuPreparedBinary<T> {
    fn validate_arguments(
        &self,
        arguments: &[PcuHostArgument<'_>],
    ) -> Result<[usize; 3], PcuCpuPreparedBinaryError> {
        super::host::validate_arguments(
            arguments,
            &self.schema[..self.argument_count],
            T::TYPE,
            T::HOST_SIZE,
        )
        .map_err(|_| PcuCpuPreparedBinaryError::InvalidArguments)?;
        let refs = [self.inputs[0].binding, self.inputs[1].binding, self.output];
        let mut positions = [0; 3];
        for (slot, target) in refs.into_iter().enumerate() {
            positions[slot] = arguments
                .iter()
                .position(|argument| argument.target() == target)
                .ok_or(PcuCpuPreparedBinaryError::InvalidArguments)?;
        }
        Ok(positions)
    }
}

// SAFETY: Full submission/profile/schema and all arithmetic are checked before mutation.
// Execution is synchronous and never retains caller storage or a pointer after return.
unsafe impl<T: PcuCheckedFloat> PcuSynchronousHostDispatchBackend<T>
    for PcuCheckedBinaryReference<T>
{
    type Error = PcuCheckedBinaryReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, T>],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        if submission
            .kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            == fusion_pcu::PcuReproducibility::PortableV1
        {
            return Err(PcuCpuPreparedBinaryError::UnsupportedProfile);
        }
        validate_dispatch_submission(submission)
            .map_err(|_| PcuCpuPreparedBinaryError::InvalidSubmission)?;
        if !parameters.is_empty() || !(2..=3).contains(&bindings.len()) {
            return Err(PcuCpuPreparedBinaryError::InvalidArguments);
        }
        let mut prepared =
            PcuCpuCheckedBinary::<T>::new().prepare_host_kernel(submission.kernel)?;
        match bindings {
            [first, second] => prepared.call(&mut [argument(first), argument(second)]),
            [first, second, third] => {
                prepared.call(&mut [argument(first), argument(second), argument(third)])
            }
            _ => Err(PcuCpuPreparedBinaryError::InvalidArguments),
        }
    }
}

const fn argument<'a, T: PcuCheckedFloat>(
    binding: &'a mut PcuHostScalarBinding<'_, T>,
) -> PcuHostArgument<'a> {
    match &mut binding.slice {
        PcuHostScalarSlice::Read(values) => PcuHostArgument::read(binding.target, values),
        PcuHostScalarSlice::ReadWrite(values) => {
            PcuHostArgument::read_write(binding.target, values)
        }
    }
}

fn fault(invocation: usize, kind: PcuExecutionFaultKind) -> PcuCpuPreparedBinaryError {
    PcuCpuPreparedBinaryError::Fault(PcuExecutionFault {
        recovered: false,
        kind,
        invocation_id: u64::try_from(invocation).expect("admitted u32 logical extent"),
    })
}

fn validate_float_profile<T: PcuCheckedFloat>(
    kernel: &PcuDispatchKernelIr<'_>,
    op: PcuDispatchFloatBinaryOp,
    underflow: PcuFloatUnderflowPolicy,
) -> Result<CheckedFloatBinaryOperandSchema, PcuCpuPreparedBinaryError> {
    match assess_checked_float_binary_operands(
        kernel,
        PcuValueType::Scalar(T::TYPE),
        op,
        underflow,
        PcuValueTypeCaps::for_scalar(T::TYPE),
    ) {
        Ok(schema) => Ok(schema),
        Err(
            CheckedFloatBinaryMapValidationError::UnsupportedInterface
            | CheckedFloatBinaryMapValidationError::UnsupportedRequirements
            | CheckedFloatBinaryMapValidationError::UnsupportedOperation(_),
        ) => Err(PcuCpuPreparedBinaryError::UnsupportedProfile),
        Err(error) => Err(PcuCpuPreparedBinaryError::InvalidKernel(error)),
    }
}

fn validated_float_region<'a>(
    kernel: &PcuDispatchKernelIr<'a>,
) -> Result<(&'a [PcuDispatchOp<'a>], u32), PcuCpuPreparedBinaryError> {
    // Only the independently qualified four-low-format Reject map opts into PortableV1.
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == fusion_pcu::PcuReproducibility::PortableV1
    {
        let description =
            fusion_pcu::describe_portable_v1_map(kernel).map_err(|error| match error {
                fusion_pcu::PcuPortableV1MapError::InvalidValueFlow(error) => {
                    PcuCpuPreparedBinaryError::InvalidValueFlow(error)
                }
                fusion_pcu::PcuPortableV1MapError::InvalidMap(error) => {
                    PcuCpuPreparedBinaryError::InvalidKernel(error)
                }
                fusion_pcu::PcuPortableV1MapError::InvalidLogicalShape(shape) => {
                    PcuCpuPreparedBinaryError::InvalidLogicalShape(shape)
                }
                _ => PcuCpuPreparedBinaryError::UnsupportedProfile,
            })?;
        if !matches!(
            description.scalar,
            fusion_pcu::PcuScalarType::F16
                | fusion_pcu::PcuScalarType::BF16
                | fusion_pcu::PcuScalarType::F8E4M3FN
                | fusion_pcu::PcuScalarType::F8E5M2
        ) {
            return Err(PcuCpuPreparedBinaryError::UnsupportedProfile);
        }
    }
    if kernel.entry.logical_shape.contains(&0) {
        return Err(PcuCpuPreparedBinaryError::InvalidLogicalShape(
            kernel.entry.logical_shape,
        ));
    }
    if kernel.entry.logical_shape[1..] != [1, 1] {
        return Err(PcuCpuPreparedBinaryError::UnsupportedProfile);
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
            return Err(PcuCpuPreparedBinaryError::InvalidKernel(
                CheckedFloatBinaryMapValidationError::MissingReturn,
            ));
        }
    };
    for (index, binding) in kernel.bindings.iter().enumerate() {
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == binding.reference())
        {
            return Err(PcuCpuPreparedBinaryError::InvalidKernel(
                CheckedFloatBinaryMapValidationError::DuplicateBinding(binding.reference()),
            ));
        }
    }
    if extent == 0 {
        return Err(PcuCpuPreparedBinaryError::InvalidGridExtent);
    }
    validate_typed_dispatch_value_flow(kernel)
        .map_err(PcuCpuPreparedBinaryError::InvalidValueFlow)?;
    Ok((body, extent))
}

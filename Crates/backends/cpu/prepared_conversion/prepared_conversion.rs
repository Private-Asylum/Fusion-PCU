//! Exact prepared F32/F64 conversions with mixed typed schemas and transactional publication.
#[rustfmt::skip]
use fusion_pcu::{
    validate_checked_float_conversion_map_kernel,
    CheckedFloatConversionMapValidationError,
    PcuBindingAccess, PcuBindingRef, PcuBindingType,
    PcuDispatchCheckedFloatConversion, PcuDispatchDataOp, PcuDispatchIndex,
    PcuDispatchKernelIr, PcuDispatchOp, PcuFloatUnderflowPolicy,
    PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel,
    PcuRangePolicy, PcuScalarType, PcuValueType, PcuValueTypeCaps,
};
use crate::PcuCpuHostError;
#[path = "execution/execution.rs"]
mod execution;

/// Explicit checked floating conversion provider; no unchecked casts are admitted.
#[derive(Debug, Clone, Copy, Default)]
pub struct PcuCpuCheckedConversion;
/// Detached mixed-width executable with frozen conversion and input broadcast layout.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuPreparedConversion {
    input: PcuBindingRef,
    output: PcuBindingRef,
    extent: usize,
    broadcast: bool,
    conversion: PcuDispatchCheckedFloatConversion,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    execute: execution::Executable,
}
impl PcuCpuPreparedConversion {
    #[must_use]
    pub const fn conversion(&self) -> PcuDispatchCheckedFloatConversion {
        self.conversion
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
        match (self.conversion, self.range) {
            (PcuDispatchCheckedFloatConversion::F32ToF64, PcuRangePolicy::Reject) => 40,
            (PcuDispatchCheckedFloatConversion::F64ToF32, PcuRangePolicy::Reject) => 41,
            (PcuDispatchCheckedFloatConversion::F32ToF64, PcuRangePolicy::Clamp) => 416,
            (PcuDispatchCheckedFloatConversion::F64ToF32, PcuRangePolicy::Clamp) => 417,
        }
    }
    const fn types(&self) -> (PcuScalarType, PcuScalarType, usize, usize) {
        match self.conversion {
            PcuDispatchCheckedFloatConversion::F32ToF64 => {
                (PcuScalarType::F32, PcuScalarType::F64, 4, 8)
            }
            PcuDispatchCheckedFloatConversion::F64ToF32 => {
                (PcuScalarType::F64, PcuScalarType::F32, 8, 4)
            }
        }
    }
}
impl PcuHostKernelBackend for PcuCpuCheckedConversion {
    type Prepared = PcuCpuPreparedConversion;
    type Error = PcuCpuHostError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let body = crate::host::validated_region(kernel)?;
        match validate_checked_float_conversion_map_kernel(
            kernel,
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        ) {
            Ok(()) => {}
            Err(
                CheckedFloatConversionMapValidationError::UnsupportedInterface
                | CheckedFloatConversionMapValidationError::UnsupportedRequirements
                | CheckedFloatConversionMapValidationError::UnsupportedOperation(_)
                | CheckedFloatConversionMapValidationError::MissingConversion,
            ) => return Err(PcuCpuHostError::UnsupportedProfile),
            Err(error) => return Err(PcuCpuHostError::Conversion(error)),
        }
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: loaded,
                binding: input,
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                result,
                value,
                conversion,
                underflow_policy,
                range_policy,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output,
                value: stored,
                ..
            }),
        ] = body
        else {
            return Err(PcuCpuHostError::UnsupportedProfile);
        };
        if loaded != value
            || result != stored
            || input == output
            || kernel.bindings.len() != 2
            || kernel.numerical_requirements.range_policy != *range_policy
        {
            return Err(PcuCpuHostError::UnsupportedProfile);
        }
        if kernel.numerical_requirements.float_underflow != *underflow_policy {
            return Err(PcuCpuHostError::HeaderUnderflowMismatch);
        }
        let prepared = PcuCpuPreparedConversion {
            input: *input,
            output: *output,
            extent: if let [PcuDispatchOp::GridStrideLoop { extent, .. }, ..] = kernel.ops {
                *extent as usize
            } else {
                kernel.entry.logical_shape[0] as usize
            },
            broadcast: *index == PcuDispatchIndex::BindingElementZero,
            conversion: *conversion,
            underflow: *underflow_policy,
            range: *range_policy,
            execute: execution::prepare(
                *conversion,
                *range_policy,
                *index == PcuDispatchIndex::BindingElementZero,
            ),
        };
        let (source, destination, _, _) = prepared.types();
        for (target, scalar, access) in [
            (*input, source, PcuBindingAccess::ReadOnly),
            (*output, destination, PcuBindingAccess::ReadWrite),
        ] {
            let binding = kernel
                .bindings
                .iter()
                .find(|binding| binding.reference() == target)
                .expect("typed SSA validates bindings");
            if binding.binding_type != PcuBindingType::Value(PcuValueType::Scalar(scalar))
                || (access == PcuBindingAccess::ReadOnly && binding.access != access)
                || (access == PcuBindingAccess::ReadWrite
                    && binding.access == PcuBindingAccess::ReadOnly)
            {
                return Err(PcuCpuHostError::UnsupportedProfile);
            }
        }
        Ok(prepared)
    }
}
impl PcuPreparedHostKernel for PcuCpuPreparedConversion {
    type Error = PcuCpuHostError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        let (source, destination, source_size, destination_size) = self.types();
        // Validate both heterogeneous schemas before accessing either complete storage span.
        if arguments.len() != 2 {
            return Err(PcuCpuHostError::Arguments(
                crate::PcuCpuHostArgumentError::Count {
                    expected: 2,
                    actual: arguments.len(),
                },
            ));
        }
        let source_slot = arguments
            .iter()
            .position(|argument| argument.target() == self.input)
            .ok_or(PcuCpuHostError::Arguments(
                crate::PcuCpuHostArgumentError::MissingBinding(self.input),
            ))?;
        let destination_slot = 1 - source_slot;
        crate::host::validate_arguments(
            &arguments[source_slot..=source_slot],
            &[(
                self.input,
                if self.broadcast { 1 } else { self.extent },
                PcuBindingAccess::ReadOnly,
            )],
            source,
            source_size,
        )
        .map_err(PcuCpuHostError::Arguments)?;
        crate::host::validate_arguments(
            &arguments[destination_slot..=destination_slot],
            &[(self.output, self.extent, PcuBindingAccess::ReadWrite)],
            destination,
            destination_size,
        )
        .map_err(PcuCpuHostError::Arguments)?;
        let (before, after) = arguments.split_at_mut(destination_slot);
        let (output, remaining) = after.split_first_mut().expect("validated output slot");
        let input = if source_slot < destination_slot {
            before[source_slot].bytes()
        } else {
            remaining[source_slot - destination_slot - 1].bytes()
        };
        (self.execute)(
            input,
            output.bytes_mut().expect("validated output access"),
            self.extent,
            self.underflow,
        )
    }
}

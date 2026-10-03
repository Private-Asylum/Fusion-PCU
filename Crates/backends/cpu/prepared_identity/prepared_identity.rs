//! Explicit dense identity and scalar broadcast, detached from IR and preserving all representation bits.
#[rustfmt::skip]
use fusion_pcu::{
    validate_scalar_identity_kernel,
    validate_scalar_broadcast_kernel,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuScalarIdentityValidationError,
    PcuScalarType,
    PcuValueType,
};
use crate::PcuCpuHostError;

/// Explicit checked-schema CPU identity provider; identity performs no numerical arithmetic.
#[derive(Debug, Clone, Copy, Default)]
pub struct PcuCpuIdentity;

/// Frozen two-binding transport; dense and one-element broadcast have separate cold admission.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuPreparedIdentity {
    input: PcuBindingRef,
    output: PcuBindingRef,
    scalar: PcuScalarType,
    broadcast: bool,
    extent: usize,
    element_size: usize,
}

impl PcuCpuPreparedIdentity {
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn element_size(&self) -> usize {
        self.element_size
    }
    #[must_use]
    pub const fn local_id(&self) -> u32 {
        if self.broadcast {
            384 + self.scalar as u32
        } else {
            64 + self.scalar as u32
        }
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

impl PcuHostKernelBackend for PcuCpuIdentity {
    type Prepared = PcuCpuPreparedIdentity;
    type Error = PcuCpuHostError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let body = crate::host::validated_region(kernel)?;
        let Some(binding) = kernel.bindings.first() else {
            return Err(PcuCpuHostError::UnsupportedProfile);
        };
        let PcuBindingType::Value(PcuValueType::Scalar(scalar)) = binding.binding_type else {
            return Err(PcuCpuHostError::UnsupportedProfile);
        };
        // These have sealed, padding-free host carriers. Subbyte/bool storage requires
        // its own host representation proof before this dynamic byte-copy route.
        let element_size = match scalar {
            PcuScalarType::I8
            | PcuScalarType::U8
            | PcuScalarType::F8E4M3FN
            | PcuScalarType::F8E5M2 => 1,
            PcuScalarType::I16 | PcuScalarType::U16 | PcuScalarType::F16 | PcuScalarType::BF16 => 2,
            PcuScalarType::I32 | PcuScalarType::U32 | PcuScalarType::F32 => 4,
            PcuScalarType::I64 | PcuScalarType::U64 | PcuScalarType::F64 => 8,
            PcuScalarType::I128 | PcuScalarType::U128 | PcuScalarType::F128 => 16,
            PcuScalarType::I256 | PcuScalarType::U256 | PcuScalarType::F256 => 32,
            PcuScalarType::I512 | PcuScalarType::U512 => 64,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4 => {
                return Err(PcuCpuHostError::UnsupportedProfile);
            }
        };
        let broadcast = matches!(
            body.first(),
            Some(PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                index: PcuDispatchIndex::BindingElementZero,
                ..
            }))
        );
        let validation = if broadcast {
            validate_scalar_broadcast_kernel(kernel, scalar)
        } else {
            validate_scalar_identity_kernel(kernel, scalar)
        };
        match validation {
            Ok(()) => {}
            Err(
                PcuScalarIdentityValidationError::UnsupportedInterface
                | PcuScalarIdentityValidationError::UnsupportedRequirements
                | PcuScalarIdentityValidationError::UnsupportedOperation(_),
            ) => return Err(PcuCpuHostError::UnsupportedProfile),
            Err(error) => return Err(PcuCpuHostError::Identity(error)),
        }
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding: input, .. }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output, ..
            }),
        ] = body
        else {
            unreachable!("validated identity has exactly load/store")
        };
        let extent = if let [PcuDispatchOp::GridStrideLoop { extent, .. }, ..] = kernel.ops {
            *extent
        } else {
            kernel.entry.logical_shape[0]
        };
        Ok(PcuCpuPreparedIdentity {
            input: *input,
            output: *output,
            scalar,
            broadcast,
            extent: extent as usize,
            element_size,
        })
    }
}
impl PcuPreparedHostKernel for PcuCpuPreparedIdentity {
    type Error = PcuCpuHostError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        crate::host::validate_arguments(
            arguments,
            &self.host_schema()[..2],
            self.scalar,
            self.element_size,
        )
        .map_err(PcuCpuHostError::Arguments)?;
        let source = arguments
            .iter()
            .position(|argument| argument.target() == self.input)
            .expect("validated identity input");
        let destination = arguments
            .iter()
            .position(|argument| argument.target() == self.output)
            .expect("validated identity output");
        let (before, after) = arguments.split_at_mut(destination);
        let (output, remaining) = after.split_first_mut().expect("validated output");
        let input = if source < destination {
            before[source].bytes()
        } else {
            remaining[source - destination - 1].bytes()
        };
        let bytes = self.extent * self.element_size;
        // Validation precedes every write. Safe host arguments retain disjoint input/output
        // borrows; copying raw initialized storage preserves signaling NaNs and wide words.
        let output = &mut output.bytes_mut().expect("validated output access")[..bytes];
        if self.broadcast {
            let value = &input[..self.element_size];
            for destination in output.chunks_exact_mut(self.element_size) {
                destination.copy_from_slice(value);
            }
        } else {
            output.copy_from_slice(&input[..bytes]);
        }
        Ok(())
    }
}

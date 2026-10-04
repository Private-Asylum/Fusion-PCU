//! Static prepared carrier/floating/integer session dispatcher; discovery and specialization remain cold.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuScalarType,
    PcuHostArgument,
    PcuPreparedHostKernel,
};
#[rustfmt::skip]
use crate::{
    MlxError,
    MlxPreparedCheckedMapHostKernel,
    MlxCheckedMapCompletion,
    MlxCheckedMapInput,
    MlxEncodedCompletion,
    MlxEncodedArray,
    MlxBinaryInput,
    MlxPreparedTransportHostKernel,
    MlxTransportCompletion,
    MlxTransportInput,
};
#[rustfmt::skip]
use super::{
    MlxPreparedHostKernel,
    MlxPreparedBinaryHostKernel,
    MlxPreparedCarrierHostKernel,
    MlxPreparedIntegerHostKernel,
    MlxPreparedDivRemHostKernel,
    MlxPreparedDivRemRoleHostKernel,
    MlxHostKernelError,
};
/// Frozen checked MLX dispatch selected once during preparation, with truthful unique input roles.
pub enum MlxPreparedDispatchKernel {
    Conversion(crate::MlxPreparedConversionHostKernel),
    Carrier(MlxPreparedCarrierHostKernel),
    Transport(MlxPreparedTransportHostKernel),
    Composed(MlxPreparedCheckedMapHostKernel),
    Unary(MlxPreparedHostKernel),
    Binary(MlxPreparedBinaryHostKernel),
    Integer(MlxPreparedIntegerHostKernel),
    DivRem(MlxPreparedDivRemHostKernel),
    DivRemRoles(MlxPreparedDivRemRoleHostKernel),
}
/// Exact frozen output cardinality; quotient and remainder never masquerade as one output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlxDispatchOutputLayout {
    Composed {
        outputs: [Option<(PcuBindingRef, usize)>; 2],
    },
    Transport {
        outputs: [Option<(PcuBindingRef, usize)>; 2],
    },
    Single {
        binding: PcuBindingRef,
        byte_len: usize,
    },
    DivRem {
        bindings: [PcuBindingRef; 2],
        byte_lengths: [usize; 2],
    },
}
/// Terminal private owners, preserving the admitted output cardinality.
pub enum MlxDispatchCompletion {
    Single(MlxEncodedCompletion),
    Transport(MlxTransportCompletion),
    Composed(MlxCheckedMapCompletion),
    DivRem([MlxEncodedArray; 2]),
}
impl MlxPreparedDispatchKernel {
    /// Every unique actual input; no unused declarations or fabricated second input resource.
    #[must_use]
    pub fn input_bindings(&self) -> &[PcuBindingRef] {
        match self {
            Self::Conversion(kernel) => kernel.input_bindings(),
            Self::Carrier(kernel) => kernel.input_bindings(),
            Self::Composed(kernel) => kernel.input_bindings(),
            Self::Transport(kernel) => kernel.plan().input_bindings(),
            Self::Unary(kernel) => std::slice::from_ref(&kernel.input),
            Self::Binary(kernel) => kernel.input_bindings(),
            Self::Integer(kernel) => kernel.input_bindings(),
            Self::DivRem(kernel) => kernel.input_bindings(),
            Self::DivRemRoles(kernel) => kernel.input_bindings(),
        }
    }
    #[must_use]
    pub const fn output_layout(&self) -> MlxDispatchOutputLayout {
        match self {
            Self::Composed(kernel) => MlxDispatchOutputLayout::Composed {
                outputs: kernel.output_layout(),
            },
            Self::Transport(kernel) => MlxDispatchOutputLayout::Transport {
                outputs: kernel.output_layout(),
            },
            Self::Conversion(kernel) => MlxDispatchOutputLayout::Single {
                binding: kernel.output_binding(),
                byte_len: kernel.output_byte_len(),
            },
            Self::Carrier(kernel) => MlxDispatchOutputLayout::Single {
                binding: kernel.output_binding(),
                byte_len: kernel.output_byte_len(),
            },
            Self::Unary(kernel) => MlxDispatchOutputLayout::Single {
                binding: kernel.output_binding(),
                byte_len: kernel.output_byte_len(),
            },
            Self::Binary(kernel) => MlxDispatchOutputLayout::Single {
                binding: kernel.output_binding(),
                byte_len: kernel.output_byte_len(),
            },
            Self::Integer(kernel) => MlxDispatchOutputLayout::Single {
                binding: kernel.output_binding(),
                byte_len: kernel.output_byte_len(),
            },
            Self::DivRem(kernel) => MlxDispatchOutputLayout::DivRem {
                bindings: *kernel.output_bindings(),
                byte_lengths: kernel.output_byte_lengths(),
            },
            Self::DivRemRoles(kernel) => MlxDispatchOutputLayout::DivRem {
                bindings: kernel.output_bindings(),
                byte_lengths: kernel.output_byte_lengths(),
            },
        }
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        match self {
            Self::Conversion(kernel) => kernel.scalar_type(),
            Self::Carrier(kernel) => kernel.scalar_type(),
            Self::Composed(kernel) => kernel.plan().value_type().scalar_type(),
            Self::Transport(kernel) => kernel.plan().scalar_type(),
            Self::Unary(kernel) => kernel.scalar_type(),
            Self::Binary(kernel) => kernel.scalar_type(),
            Self::Integer(kernel) => kernel.scalar_type(),
            Self::DivRem(kernel) => kernel.scalar_type(),
            Self::DivRemRoles(kernel) => kernel.scalar_type(),
        }
    }
    /// Exact input type for one actual resource; conversion outputs can have a different type.
    #[must_use]
    pub fn input_scalar_type(&self, binding: PcuBindingRef) -> Option<PcuScalarType> {
        if !self.input_bindings().contains(&binding) {
            return None;
        }
        Some(match self {
            Self::Conversion(kernel) => kernel.source_scalar_type(),
            _ => self.scalar_type(),
        })
    }
    /// Safe unique-input byte/resident seam; variants retain their real arity and exact schema.
    ///
    /// # Errors
    /// Returns type, extent, binding or affinity mismatch before GPU work, or checked/native failure.
    pub fn execute_inputs(
        &mut self,
        inputs: &[MlxBinaryInput<'_>],
    ) -> Result<MlxDispatchCompletion, MlxError> {
        match self {
            Self::Composed(kernel) => {
                return composed_inputs(kernel, inputs).map(MlxDispatchCompletion::Composed);
            }
            Self::Transport(kernel) => {
                return transport_inputs(kernel, inputs).map(MlxDispatchCompletion::Transport);
            }
            Self::Carrier(kernel) => {
                kernel.reset_write_fact();
                let [input] = inputs else {
                    return Err(MlxError::InvalidExtent);
                };
                match input {
                    MlxBinaryInput::HostBytes {
                        target,
                        scalar,
                        bytes,
                    } if *target == kernel.input_binding() => {
                        kernel.execute_encoded_bytes(*scalar, bytes)
                    }
                    MlxBinaryInput::Resident { target, array }
                        if *target == kernel.input_binding() =>
                    {
                        kernel.execute_resident(array)
                    }
                    _ => Err(MlxError::InvalidRequest(
                        "unexpected MLX carrier input binding".into(),
                    )),
                }
            }
            Self::Conversion(kernel) => kernel.execute_inputs(inputs),
            Self::Binary(kernel) => kernel.execute_inputs(inputs),
            Self::Integer(kernel) => kernel.execute_inputs(inputs),
            Self::DivRem(kernel) => {
                return kernel
                    .execute_inputs(inputs)
                    .map(MlxDispatchCompletion::DivRem);
            }
            Self::DivRemRoles(kernel) => {
                return kernel
                    .execute_inputs(inputs)
                    .map(MlxDispatchCompletion::DivRem);
            }
            Self::Unary(kernel) => {
                kernel.native.reset_write_fact();
                let [input] = inputs else {
                    return Err(MlxError::InvalidExtent);
                };
                match input {
                    MlxBinaryInput::HostBytes {
                        target,
                        scalar,
                        bytes,
                    } => {
                        if *target != kernel.input_binding() {
                            return Err(MlxError::InvalidRequest(
                                "unexpected MLX unary input binding".into(),
                            ));
                        }
                        kernel.execute_encoded_bytes(*scalar, bytes)
                    }
                    MlxBinaryInput::Resident { target, array } => {
                        if *target != kernel.input_binding() {
                            return Err(MlxError::InvalidRequest(
                                "unexpected MLX unary input binding".into(),
                            ));
                        }
                        kernel.execute_resident(array)
                    }
                }
            }
        }
        .map(MlxDispatchCompletion::Single)
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        match self {
            Self::Conversion(kernel) => kernel.last_call_may_have_written(),
            Self::Carrier(kernel) => kernel.last_call_may_have_written(),
            Self::Composed(kernel) => kernel.last_call_may_have_written(),
            Self::Transport(kernel) => kernel.last_call_may_have_written(),
            Self::Unary(kernel) => kernel.last_call_may_have_written(),
            Self::Binary(kernel) => kernel.last_call_may_have_written(),
            Self::Integer(kernel) => kernel.last_call_may_have_written(),
            Self::DivRem(kernel) => kernel.last_call_may_have_written(),
            Self::DivRemRoles(kernel) => kernel.last_call_may_have_written(),
        }
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        match self {
            Self::Conversion(kernel) => kernel.last_call_completion_uncertain(),
            Self::Carrier(kernel) => kernel.last_call_completion_uncertain(),
            Self::Composed(kernel) => kernel.last_call_completion_uncertain(),
            Self::Transport(kernel) => kernel.last_call_completion_uncertain(),
            Self::Unary(kernel) => kernel.last_call_completion_uncertain(),
            Self::Binary(kernel) => kernel.last_call_completion_uncertain(),
            Self::Integer(kernel) => kernel.last_call_completion_uncertain(),
            Self::DivRem(kernel) => kernel.last_call_completion_uncertain(),
            Self::DivRemRoles(kernel) => kernel.last_call_completion_uncertain(),
        }
    }
}
impl PcuPreparedHostKernel for MlxPreparedDispatchKernel {
    type Error = MlxHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        match self {
            Self::Conversion(kernel) => kernel
                .call(arguments)
                .map_err(fusion_pcu::PcuHostDispatchError::Backend),
            Self::Carrier(kernel) => kernel.call(arguments),
            Self::Composed(kernel) => kernel.call(arguments),
            Self::Transport(kernel) => kernel.call(arguments),
            Self::Unary(kernel) => kernel.call(arguments),
            Self::Binary(kernel) => kernel.call(arguments),
            Self::Integer(kernel) => kernel.call(arguments),
            Self::DivRem(kernel) => kernel.call(arguments),
            Self::DivRemRoles(kernel) => kernel.call(arguments),
        }
    }
}

fn transport_inputs(
    kernel: &mut MlxPreparedTransportHostKernel,
    inputs: &[MlxBinaryInput<'_>],
) -> Result<MlxTransportCompletion, MlxError> {
    if inputs.is_empty() || inputs.len() > 4 {
        return Err(MlxError::InvalidExtent);
    }
    let mut actual = [transport_input(inputs[0]); 4];
    for (slot, input) in inputs.iter().enumerate() {
        actual[slot] = transport_input(*input);
    }
    kernel
        .execute_inputs(&actual[..inputs.len()])
        .map_err(|error| match error {
            fusion_pcu::PcuHostDispatchError::Backend(error) => error,
            error => MlxError::InvalidRequest(format!("invalid transport input: {error:?}")),
        })
}
const fn transport_input(input: MlxBinaryInput<'_>) -> MlxTransportInput<'_> {
    match input {
        MlxBinaryInput::HostBytes {
            target,
            scalar,
            bytes,
        } => MlxTransportInput::HostBytes {
            target,
            scalar,
            bytes,
        },
        MlxBinaryInput::Resident { target, array } => MlxTransportInput::Resident { target, array },
    }
}

fn composed_inputs(
    kernel: &mut MlxPreparedCheckedMapHostKernel,
    inputs: &[MlxBinaryInput<'_>],
) -> Result<MlxCheckedMapCompletion, MlxError> {
    if inputs.is_empty() || inputs.len() > 4 {
        return Err(MlxError::InvalidExtent);
    }
    let mut actual = [composed_input(inputs[0]); 4];
    for (slot, input) in inputs.iter().enumerate() {
        actual[slot] = composed_input(*input);
    }
    kernel
        .execute_inputs(&actual[..inputs.len()])
        .map_err(|error| match error {
            fusion_pcu::PcuHostDispatchError::Backend(error) => error,
            error => MlxError::InvalidRequest(format!("invalid composed input: {error:?}")),
        })
}
const fn composed_input(input: MlxBinaryInput<'_>) -> MlxCheckedMapInput<'_> {
    match input {
        MlxBinaryInput::HostBytes {
            target,
            scalar,
            bytes,
        } => MlxCheckedMapInput::HostBytes {
            target,
            scalar,
            bytes,
        },
        MlxBinaryInput::Resident { target, array } => {
            MlxCheckedMapInput::Resident { target, array }
        }
    }
}

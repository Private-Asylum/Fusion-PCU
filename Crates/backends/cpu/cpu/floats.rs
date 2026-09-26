use fusion_pcu::{
    validate_f64_map_kernel,
    validate_host_scalar_bindings,
    PcuBindingRef,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuDispatchOpCaps,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuParameterValue,
    PcuSynchronousHostDispatchBackend,
};

use crate::{
    validation,
    VALUE_SLOTS,
};

/// Failure to interpret the bounded `f32` map profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuF32ReferenceError {
    InvalidSubmission,
    UnsupportedInstruction(usize),
    InvalidValue(PcuDispatchValueId),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
    MissingReturn,
    MissingStore,
}

/// CPU oracle for the bounded `f32` indexed-map profile.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuF32Reference;

impl PcuF32Reference {
    /// Instruction floor implemented by the bounded scalar `f32` CPU oracle.
    ///
    /// Backends that admit kernels by capability should report both `BINDING_LOAD` and
    /// `BINDING_LOAD_ELEMENT_ZERO` for kernels that broadcast a one-element resource.
    pub const SUPPORTED_INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::VALUE_CONSTANT
        .union(PcuDispatchOpCaps::ALU_ADD)
        .union(PcuDispatchOpCaps::ALU_SUB)
        .union(PcuDispatchOpCaps::ALU_MUL)
        .union(PcuDispatchOpCaps::ALU_DIV)
        .union(PcuDispatchOpCaps::ALU_MIN)
        .union(PcuDispatchOpCaps::ALU_MAX)
        .union(PcuDispatchOpCaps::CONTROL_RETURN)
        .union(PcuDispatchOpCaps::CONTROL_LOOP)
        .union(PcuDispatchOpCaps::BINDING_LOAD)
        .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
        .union(PcuDispatchOpCaps::BINDING_STORE);
}

// SAFETY: The reference executes on the calling thread, never retains slice references or raw
// pointers, and returns only after its final read/write has completed, including every error path.
unsafe impl PcuSynchronousHostDispatchBackend<f32> for PcuF32Reference {
    type Error = PcuF32ReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, f32>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<f32, ()>(submission, bindings)
            .map_err(|_| PcuF32ReferenceError::InvalidSubmission)?;
        validation::validate_program(submission, bindings)?;
        for invocation in 0..submission.shape.invocation_count().get() as usize {
            let mut values = [None; VALUE_SLOTS];
            for op in submission.kernel.ops {
                match op {
                    PcuDispatchOp::GridStrideLoop { extent, body } => {
                        let stride = submission.shape.invocation_count().get() as usize;
                        let mut logical = invocation;
                        while logical < *extent as usize {
                            validation::execute_grid_stride_body(body, logical, bindings)?;
                            if (*extent as usize - logical) <= stride {
                                break;
                            }
                            logical += stride;
                        }
                    }
                    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                        result,
                        binding,
                        index,
                    }) => {
                        let source = bindings
                            .iter()
                            .find(|candidate| candidate.target == *binding)
                            .ok_or(PcuF32ReferenceError::MissingBinding(*binding))?;
                        let element = if *index == PcuDispatchIndex::BindingElementZero {
                            0
                        } else {
                            invocation
                        };
                        let value = match &source.slice {
                            PcuHostScalarSlice::Read(slice) => slice[element],
                            PcuHostScalarSlice::ReadWrite(slice) => slice[element],
                        };
                        values[usize::from(result.0)] = Some(value);
                    }
                    PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                        let PcuParameterValue::F32(bits) = value else {
                            unreachable!("preflight checks constants");
                        };
                        values[usize::from(result.0)] = Some(f32::from_bits(*bits));
                    }
                    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                        result,
                        op,
                        lhs,
                        rhs,
                        ..
                    }) => {
                        let left = values[usize::from(lhs.0)]
                            .ok_or(PcuF32ReferenceError::InvalidValue(*lhs))?;
                        let right = values[usize::from(rhs.0)]
                            .ok_or(PcuF32ReferenceError::InvalidValue(*rhs))?;
                        let value = match op {
                            PcuDispatchAluOp::Add => left + right,
                            PcuDispatchAluOp::Sub => left - right,
                            PcuDispatchAluOp::Mul => left * right,
                            PcuDispatchAluOp::Div => left / right,
                            PcuDispatchAluOp::Min => validation::f32_min(left, right),
                            PcuDispatchAluOp::Max => validation::f32_max(left, right),
                            _ => unreachable!("preflight checks arithmetic"),
                        };
                        values[usize::from(result.0)] = Some(value);
                    }
                    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                        binding, value, ..
                    }) => {
                        let destination = bindings
                            .iter_mut()
                            .find(|candidate| candidate.target == *binding)
                            .ok_or(PcuF32ReferenceError::MissingBinding(*binding))?;
                        let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice else {
                            return Err(PcuF32ReferenceError::AccessMismatch(*binding));
                        };
                        slice[invocation] = values[usize::from(value.0)]
                            .ok_or(PcuF32ReferenceError::InvalidValue(*value))?;
                    }
                    PcuDispatchOp::Control(PcuDispatchControlOp::Return) => break,
                    _ => unreachable!("preflight checks instructions"),
                }
            }
        }
        Ok(())
    }
}

/// CPU oracle for direct and grid-stride f64 Add/Sub/Mul/Div maps.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuF64Reference;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuF64ReferenceError {
    InvalidSubmission,
    InvalidKernel,
    MissingBinding(PcuBindingRef),
    MissingValue(PcuDispatchValueId),
    AccessMismatch(PcuBindingRef),
}

// SAFETY: Execution is synchronous and retains no host slice references after returning.
unsafe impl PcuSynchronousHostDispatchBackend<f64> for PcuF64Reference {
    type Error = PcuF64ReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, f64>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<f64, ()>(submission, bindings)
            .map_err(|_| PcuF64ReferenceError::InvalidSubmission)?;
        validate_f64_map_kernel(submission.kernel)
            .map_err(|_| PcuF64ReferenceError::InvalidKernel)?;
        let width = submission.shape.invocation_count().get() as usize;
        if let Some(PcuDispatchOp::GridStrideLoop { extent, body }) = submission.kernel.ops.first()
        {
            for invocation in 0..width {
                let mut logical = invocation;
                while logical < *extent as usize {
                    execute_f64_map_ops(body, logical, bindings)?;
                    if (*extent as usize - logical) <= width {
                        break;
                    }
                    logical += width;
                }
            }
        } else {
            for invocation in 0..width {
                execute_f64_map_ops(submission.kernel.ops, invocation, bindings)?;
            }
        }
        Ok(())
    }
}

fn execute_f64_map_ops(
    ops: &[PcuDispatchOp<'_>],
    logical: usize,
    bindings: &mut [PcuHostScalarBinding<'_, f64>],
) -> Result<(), PcuF64ReferenceError> {
    let mut values = [None; VALUE_SLOTS];
    for op in ops {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                let source = bindings
                    .iter()
                    .find(|candidate| candidate.target == *binding)
                    .ok_or(PcuF64ReferenceError::MissingBinding(*binding))?;
                let element = if *index == PcuDispatchIndex::BindingElementZero {
                    0
                } else {
                    logical
                };
                let value = match &source.slice {
                    PcuHostScalarSlice::Read(slice) => slice[element],
                    PcuHostScalarSlice::ReadWrite(slice) => slice[element],
                };
                values[usize::from(result.0)] = Some(value);
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
                ..
            }) => {
                let left =
                    values[usize::from(lhs.0)].ok_or(PcuF64ReferenceError::MissingValue(*lhs))?;
                let right =
                    values[usize::from(rhs.0)].ok_or(PcuF64ReferenceError::MissingValue(*rhs))?;
                values[usize::from(result.0)] = Some(match op {
                    PcuDispatchAluOp::Add => left + right,
                    PcuDispatchAluOp::Sub => left - right,
                    PcuDispatchAluOp::Mul => left * right,
                    PcuDispatchAluOp::Div => left / right,
                    _ => unreachable!("validator limits f64 arithmetic"),
                });
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
                let destination = bindings
                    .iter_mut()
                    .find(|candidate| candidate.target == *binding)
                    .ok_or(PcuF64ReferenceError::MissingBinding(*binding))?;
                let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice else {
                    return Err(PcuF64ReferenceError::AccessMismatch(*binding));
                };
                slice[logical] = values[usize::from(value.0)]
                    .ok_or(PcuF64ReferenceError::MissingValue(*value))?;
            }
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) => break,
            _ => unreachable!("validator limits f64 indexed map operations"),
        }
    }
    Ok(())
}

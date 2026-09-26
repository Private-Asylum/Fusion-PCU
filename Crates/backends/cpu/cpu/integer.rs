use fusion_pcu::{
    validate_host_scalar_bindings,
    validate_i16_map_kernel,
    validate_i32_checked_div_rem_kernel,
    validate_i32_map_kernel,
    validate_i64_map_kernel,
    validate_i8_map_kernel,
    validate_u16_map_kernel,
    validate_u32_checked_div_rem_kernel,
    validate_u32_map_kernel,
    validate_u64_checked_div_rem_kernel,
    validate_u64_identity_kernel,
    validate_u64_map_kernel,
    validate_u8_map_kernel,
    PcuBindingRef,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuSynchronousHostDispatchBackend,
};

use crate::VALUE_SLOTS;

/// Failure to execute the typed wrapping `u8` map reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuU8MapReferenceError {
    InvalidSubmission,
    InvalidKernel(fusion_pcu::PcuU8MapValidationError),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
}

/// CPU oracle for the bounded typed `u8` indexed arithmetic map.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuU8MapReference;

// SAFETY: Executes synchronously on caller-owned slices and retains no references after return.
unsafe impl PcuSynchronousHostDispatchBackend<u8> for PcuU8MapReference {
    type Error = PcuU8MapReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, u8>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<u8, ()>(submission, bindings)
            .map_err(|_| PcuU8MapReferenceError::InvalidSubmission)?;
        validate_u8_map_kernel(submission.kernel).map_err(PcuU8MapReferenceError::InvalidKernel)?;
        let (body, extent, grid) = match submission.kernel.ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (*body, *extent as usize, true),
            ops => (
                ops,
                submission.shape.invocation_count().get() as usize,
                false,
            ),
        };
        let stride = submission.shape.invocation_count().get() as usize;
        for invocation in 0..stride {
            let mut logical = invocation;
            while logical < extent {
                let mut values = [0_u8; VALUE_SLOTS];
                for op in body {
                    match op {
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result,
                            binding,
                            ..
                        }) => {
                            let source = bindings
                                .iter()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuU8MapReferenceError::MissingBinding(*binding))?;
                            let value = match &source.slice {
                                PcuHostScalarSlice::Read(slice) => slice[logical],
                                PcuHostScalarSlice::ReadWrite(slice) => slice[logical],
                            };
                            values[usize::from(result.0)] = value;
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            result,
                            op,
                            lhs,
                            rhs,
                            ..
                        }) => {
                            let a = values[usize::from(lhs.0)];
                            let b = values[usize::from(rhs.0)];
                            values[usize::from(result.0)] = match op {
                                PcuDispatchAluOp::Add => a.wrapping_add(b),
                                PcuDispatchAluOp::Sub => a.wrapping_sub(b),
                                PcuDispatchAluOp::Mul => a.wrapping_mul(b),
                                _ => unreachable!("validator admits only wrapping Add/Sub/Mul"),
                            };
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding,
                            value,
                            ..
                        }) => {
                            let destination = bindings
                                .iter_mut()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuU8MapReferenceError::MissingBinding(*binding))?;
                            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice
                            else {
                                return Err(PcuU8MapReferenceError::AccessMismatch(*binding));
                            };
                            slice[logical] = values[usize::from(value.0)];
                        }
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {}
                        _ => unreachable!("validator admits only map operations"),
                    }
                }
                if !grid {
                    break;
                }
                if extent - logical <= stride {
                    break;
                }
                logical += stride;
            }
        }
        Ok(())
    }
}

/// Failure to execute the typed wrapping `u16` map reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuU16MapReferenceError {
    InvalidSubmission,
    InvalidKernel(fusion_pcu::PcuU16MapValidationError),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
}

/// CPU oracle for the bounded typed `u16` indexed arithmetic map.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuU16MapReference;

// SAFETY: Executes synchronously on caller-owned slices and retains no references after return.
unsafe impl PcuSynchronousHostDispatchBackend<u16> for PcuU16MapReference {
    type Error = PcuU16MapReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, u16>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<u16, ()>(submission, bindings)
            .map_err(|_| PcuU16MapReferenceError::InvalidSubmission)?;
        validate_u16_map_kernel(submission.kernel)
            .map_err(PcuU16MapReferenceError::InvalidKernel)?;
        let (body, extent, grid) = match submission.kernel.ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (*body, *extent as usize, true),
            ops => (
                ops,
                submission.shape.invocation_count().get() as usize,
                false,
            ),
        };
        let stride = submission.shape.invocation_count().get() as usize;
        for invocation in 0..stride {
            let mut logical = invocation;
            while logical < extent {
                let mut values = [0_u16; VALUE_SLOTS];
                for op in body {
                    match op {
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result,
                            binding,
                            ..
                        }) => {
                            let source = bindings
                                .iter()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuU16MapReferenceError::MissingBinding(*binding))?;
                            let value = match &source.slice {
                                PcuHostScalarSlice::Read(slice) => slice[logical],
                                PcuHostScalarSlice::ReadWrite(slice) => slice[logical],
                            };
                            values[usize::from(result.0)] = value;
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            result,
                            op,
                            lhs,
                            rhs,
                            ..
                        }) => {
                            let a = values[usize::from(lhs.0)];
                            let b = values[usize::from(rhs.0)];
                            values[usize::from(result.0)] = match op {
                                PcuDispatchAluOp::Add => a.wrapping_add(b),
                                PcuDispatchAluOp::Sub => a.wrapping_sub(b),
                                PcuDispatchAluOp::Mul => a.wrapping_mul(b),
                                _ => unreachable!("validator admits only wrapping Add/Sub/Mul"),
                            };
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding,
                            value,
                            ..
                        }) => {
                            let destination = bindings
                                .iter_mut()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuU16MapReferenceError::MissingBinding(*binding))?;
                            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice
                            else {
                                return Err(PcuU16MapReferenceError::AccessMismatch(*binding));
                            };
                            slice[logical] = values[usize::from(value.0)];
                        }
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {}
                        _ => unreachable!("validator admits only map operations"),
                    }
                }
                if !grid {
                    break;
                }
                if extent - logical <= stride {
                    break;
                }
                logical += stride;
            }
        }
        Ok(())
    }
}

/// Failure to execute the typed wrapping `u32` map reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuU32MapReferenceError {
    InvalidSubmission,
    InvalidKernel(fusion_pcu::PcuU32MapValidationError),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
    /// A checked arithmetic domain fault. Output buffers are unusable after this error.
    Fault(PcuExecutionFault),
}

/// CPU oracle for the bounded typed `u32` indexed arithmetic map.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuU32MapReference;

// SAFETY: Executes synchronously on caller-owned slices and retains no references after return.
unsafe impl PcuSynchronousHostDispatchBackend<u32> for PcuU32MapReference {
    type Error = PcuU32MapReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, u32>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<u32, ()>(submission, bindings)
            .map_err(|_| PcuU32MapReferenceError::InvalidSubmission)?;
        if validate_u32_checked_div_rem_kernel(submission.kernel).is_ok() {
            return run_checked_u32_div_rem(submission, bindings);
        }
        validate_u32_map_kernel(submission.kernel)
            .map_err(PcuU32MapReferenceError::InvalidKernel)?;
        let (body, extent, grid) = match submission.kernel.ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (*body, *extent as usize, true),
            ops => (
                ops,
                submission.shape.invocation_count().get() as usize,
                false,
            ),
        };
        let stride = submission.shape.invocation_count().get() as usize;
        for invocation in 0..stride {
            let mut logical = invocation;
            while logical < extent {
                let mut values = [0_u32; VALUE_SLOTS];
                for op in body {
                    match op {
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result,
                            binding,
                            ..
                        }) => {
                            let source = bindings
                                .iter()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuU32MapReferenceError::MissingBinding(*binding))?;
                            let value = match &source.slice {
                                PcuHostScalarSlice::Read(slice) => slice[logical],
                                PcuHostScalarSlice::ReadWrite(slice) => slice[logical],
                            };
                            values[usize::from(result.0)] = value;
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            result,
                            op,
                            lhs,
                            rhs,
                            ..
                        }) => {
                            let a = values[usize::from(lhs.0)];
                            let b = values[usize::from(rhs.0)];
                            values[usize::from(result.0)] = match op {
                                PcuDispatchAluOp::Add => a.wrapping_add(b),
                                PcuDispatchAluOp::Sub => a.wrapping_sub(b),
                                PcuDispatchAluOp::Mul => a.wrapping_mul(b),
                                _ => unreachable!("validator admits only wrapping Add/Sub/Mul"),
                            };
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding,
                            value,
                            ..
                        }) => {
                            let destination = bindings
                                .iter_mut()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuU32MapReferenceError::MissingBinding(*binding))?;
                            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice
                            else {
                                return Err(PcuU32MapReferenceError::AccessMismatch(*binding));
                            };
                            slice[logical] = values[usize::from(value.0)];
                        }
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {}
                        _ => unreachable!("validator admits only map operations"),
                    }
                }
                if !grid {
                    break;
                }
                if extent - logical <= stride {
                    break;
                }
                logical += stride;
            }
        }
        Ok(())
    }
}

fn run_checked_u32_div_rem(
    submission: PcuDispatchSubmission<'_>,
    bindings: &mut [PcuHostScalarBinding<'_, u32>],
) -> Result<(), PcuU32MapReferenceError> {
    let extent = match submission.kernel.ops {
        [PcuDispatchOp::GridStrideLoop { extent, .. }, _] => *extent as usize,
        _ => submission.shape.invocation_count().get() as usize,
    };
    let [input_a, input_b, quotient, remainder] = [
        submission.kernel.bindings[0].reference(),
        submission.kernel.bindings[1].reference(),
        submission.kernel.bindings[2].reference(),
        submission.kernel.bindings[3].reference(),
    ];
    for logical in 0..extent {
        let read = |target| {
            bindings
                .iter()
                .find(|candidate| candidate.target == target)
                .and_then(|candidate| match &candidate.slice {
                    PcuHostScalarSlice::Read(slice) => slice.get(logical).copied(),
                    PcuHostScalarSlice::ReadWrite(slice) => slice.get(logical).copied(),
                })
        };
        let lhs = read(input_a).ok_or(PcuU32MapReferenceError::MissingBinding(input_a))?;
        let rhs = read(input_b).ok_or(PcuU32MapReferenceError::MissingBinding(input_b))?;
        if rhs == 0 {
            return Err(PcuU32MapReferenceError::Fault(PcuExecutionFault {
                kind: PcuExecutionFaultKind::DivideByZero,
                invocation_id: u64::try_from(logical).unwrap_or(u64::MAX),
            }));
        }
        let quotient_value = lhs / rhs;
        let remainder_value = lhs % rhs;
        for (target, value) in [(quotient, quotient_value), (remainder, remainder_value)] {
            let destination = bindings
                .iter_mut()
                .find(|candidate| candidate.target == target)
                .ok_or(PcuU32MapReferenceError::MissingBinding(target))?;
            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice else {
                return Err(PcuU32MapReferenceError::AccessMismatch(target));
            };
            let Some(output) = slice.get_mut(logical) else {
                return Err(PcuU32MapReferenceError::InvalidSubmission);
            };
            *output = value;
        }
    }
    Ok(())
}

/// Failure to execute the typed wrapping `u64` map reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuU64MapReferenceError {
    InvalidSubmission,
    InvalidKernel(fusion_pcu::PcuU64MapValidationError),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
    /// A checked arithmetic domain fault. Both output buffers are unusable after this error.
    Fault(PcuExecutionFault),
}

/// CPU oracle for the bounded typed `u64` indexed arithmetic map.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuU64MapReference;

// SAFETY: Executes synchronously on caller-owned slices and retains no references after return.
unsafe impl PcuSynchronousHostDispatchBackend<u64> for PcuU64MapReference {
    type Error = PcuU64MapReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, u64>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<u64, ()>(submission, bindings)
            .map_err(|_| PcuU64MapReferenceError::InvalidSubmission)?;
        if validate_u64_checked_div_rem_kernel(submission.kernel).is_ok() {
            return run_checked_u64_div_rem(submission, bindings);
        }
        let identity = if validate_u64_identity_kernel(submission.kernel).is_ok() {
            true
        } else {
            validate_u64_map_kernel(submission.kernel)
                .map_err(PcuU64MapReferenceError::InvalidKernel)?;
            false
        };
        let (body, extent, grid) = match submission.kernel.ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (*body, *extent as usize, true),
            ops => (
                ops,
                submission.shape.invocation_count().get() as usize,
                false,
            ),
        };
        let stride = submission.shape.invocation_count().get() as usize;
        for invocation in 0..stride {
            let mut logical = invocation;
            while logical < extent {
                let mut values = [0_u64; VALUE_SLOTS];
                for op in body {
                    match op {
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result,
                            binding,
                            ..
                        }) => {
                            let source = bindings
                                .iter()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuU64MapReferenceError::MissingBinding(*binding))?;
                            let value = match &source.slice {
                                PcuHostScalarSlice::Read(slice) => slice[logical],
                                PcuHostScalarSlice::ReadWrite(slice) => slice[logical],
                            };
                            values[usize::from(result.0)] = value;
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            result,
                            op,
                            lhs,
                            rhs,
                            ..
                        }) => {
                            let a = values[usize::from(lhs.0)];
                            let b = values[usize::from(rhs.0)];
                            values[usize::from(result.0)] = match op {
                                PcuDispatchAluOp::Add => a.wrapping_add(b),
                                PcuDispatchAluOp::Sub => a.wrapping_sub(b),
                                PcuDispatchAluOp::Mul => a.wrapping_mul(b),
                                _ => unreachable!("validator admits only wrapping Add/Sub/Mul"),
                            };
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding,
                            value,
                            ..
                        }) => {
                            let destination = bindings
                                .iter_mut()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuU64MapReferenceError::MissingBinding(*binding))?;
                            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice
                            else {
                                return Err(PcuU64MapReferenceError::AccessMismatch(*binding));
                            };
                            slice[logical] = values[usize::from(value.0)];
                        }
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {}
                        _ if identity => {
                            unreachable!("identity validator admits only copy operations")
                        }
                        _ => unreachable!("validator admits only map operations"),
                    }
                }
                if !grid {
                    break;
                }
                if extent - logical <= stride {
                    break;
                }
                logical += stride;
            }
        }
        Ok(())
    }
}

fn run_checked_u64_div_rem(
    submission: PcuDispatchSubmission<'_>,
    bindings: &mut [PcuHostScalarBinding<'_, u64>],
) -> Result<(), PcuU64MapReferenceError> {
    let extent = match submission.kernel.ops {
        [PcuDispatchOp::GridStrideLoop { extent, .. }, _] => *extent as usize,
        _ => submission.shape.invocation_count().get() as usize,
    };
    let [input_a, input_b, quotient, remainder] = [
        submission.kernel.bindings[0].reference(),
        submission.kernel.bindings[1].reference(),
        submission.kernel.bindings[2].reference(),
        submission.kernel.bindings[3].reference(),
    ];
    // Validate every divisor before mutating either output, and report the lowest logical id.
    for logical in 0..extent {
        let rhs = read_u64_binding(bindings, input_b, logical)
            .ok_or(PcuU64MapReferenceError::MissingBinding(input_b))?;
        if rhs == 0 {
            return Err(PcuU64MapReferenceError::Fault(PcuExecutionFault {
                kind: PcuExecutionFaultKind::DivideByZero,
                invocation_id: u64::try_from(logical).unwrap_or(u64::MAX),
            }));
        }
    }
    for logical in 0..extent {
        let lhs = read_u64_binding(bindings, input_a, logical)
            .ok_or(PcuU64MapReferenceError::MissingBinding(input_a))?;
        let rhs = read_u64_binding(bindings, input_b, logical)
            .ok_or(PcuU64MapReferenceError::MissingBinding(input_b))?;
        for (target, value) in [(quotient, lhs / rhs), (remainder, lhs % rhs)] {
            let destination = bindings
                .iter_mut()
                .find(|candidate| candidate.target == target)
                .ok_or(PcuU64MapReferenceError::MissingBinding(target))?;
            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice else {
                return Err(PcuU64MapReferenceError::AccessMismatch(target));
            };
            let Some(output) = slice.get_mut(logical) else {
                return Err(PcuU64MapReferenceError::InvalidSubmission);
            };
            *output = value;
        }
    }
    Ok(())
}

fn read_u64_binding(
    bindings: &[PcuHostScalarBinding<'_, u64>],
    target: PcuBindingRef,
    logical: usize,
) -> Option<u64> {
    bindings
        .iter()
        .find(|candidate| candidate.target == target)
        .and_then(|candidate| match &candidate.slice {
            PcuHostScalarSlice::Read(slice) => slice.get(logical).copied(),
            PcuHostScalarSlice::ReadWrite(slice) => slice.get(logical).copied(),
        })
}

/// Failure to execute the typed wrapping `i64` map reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuI64MapReferenceError {
    InvalidSubmission,
    InvalidKernel(fusion_pcu::PcuI64MapValidationError),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
}

/// CPU oracle for the bounded typed `i64` indexed arithmetic map.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuI64MapReference;

// SAFETY: Executes synchronously on caller-owned slices and retains no references after return.
unsafe impl PcuSynchronousHostDispatchBackend<i64> for PcuI64MapReference {
    type Error = PcuI64MapReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, i64>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<i64, ()>(submission, bindings)
            .map_err(|_| PcuI64MapReferenceError::InvalidSubmission)?;
        validate_i64_map_kernel(submission.kernel)
            .map_err(PcuI64MapReferenceError::InvalidKernel)?;
        let (body, extent, grid) = match submission.kernel.ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (*body, *extent as usize, true),
            ops => (
                ops,
                submission.shape.invocation_count().get() as usize,
                false,
            ),
        };
        let stride = submission.shape.invocation_count().get() as usize;
        for invocation in 0..stride {
            let mut logical = invocation;
            while logical < extent {
                let mut values = [0_i64; VALUE_SLOTS];
                for op in body {
                    match op {
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result,
                            binding,
                            ..
                        }) => {
                            let source = bindings
                                .iter()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuI64MapReferenceError::MissingBinding(*binding))?;
                            let value = match &source.slice {
                                PcuHostScalarSlice::Read(slice) => slice[logical],
                                PcuHostScalarSlice::ReadWrite(slice) => slice[logical],
                            };
                            values[usize::from(result.0)] = value;
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            result,
                            op,
                            lhs,
                            rhs,
                            ..
                        }) => {
                            let a = values[usize::from(lhs.0)];
                            let b = values[usize::from(rhs.0)];
                            values[usize::from(result.0)] = match op {
                                PcuDispatchAluOp::Add => a.wrapping_add(b),
                                PcuDispatchAluOp::Sub => a.wrapping_sub(b),
                                PcuDispatchAluOp::Mul => a.wrapping_mul(b),
                                _ => unreachable!("validator admits only wrapping Add/Sub/Mul"),
                            };
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding,
                            value,
                            ..
                        }) => {
                            let destination = bindings
                                .iter_mut()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuI64MapReferenceError::MissingBinding(*binding))?;
                            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice
                            else {
                                return Err(PcuI64MapReferenceError::AccessMismatch(*binding));
                            };
                            slice[logical] = values[usize::from(value.0)];
                        }
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {}
                        _ => unreachable!("validator admits only i64 map operations"),
                    }
                }
                if !grid || extent - logical <= stride {
                    break;
                }
                logical += stride;
            }
        }
        Ok(())
    }
}

/// Failure to execute the typed wrapping `i32` map reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuI32MapReferenceError {
    InvalidSubmission,
    InvalidKernel(fusion_pcu::PcuI32MapValidationError),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
    /// A checked arithmetic domain fault. Both output buffers are unusable after this error.
    Fault(PcuExecutionFault),
}

/// CPU oracle for the bounded typed `i32` indexed arithmetic map.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuI32MapReference;

// SAFETY: Executes synchronously on caller-owned slices and retains no references after return.
unsafe impl PcuSynchronousHostDispatchBackend<i32> for PcuI32MapReference {
    type Error = PcuI32MapReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, i32>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<i32, ()>(submission, bindings)
            .map_err(|_| PcuI32MapReferenceError::InvalidSubmission)?;
        if validate_i32_checked_div_rem_kernel(submission.kernel).is_ok() {
            return run_checked_i32_div_rem(submission, bindings);
        }
        validate_i32_map_kernel(submission.kernel)
            .map_err(PcuI32MapReferenceError::InvalidKernel)?;
        let (body, extent, grid) = match submission.kernel.ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (*body, *extent as usize, true),
            ops => (
                ops,
                submission.shape.invocation_count().get() as usize,
                false,
            ),
        };
        let stride = submission.shape.invocation_count().get() as usize;
        for invocation in 0..stride {
            let mut logical = invocation;
            while logical < extent {
                let mut values = [0_i32; VALUE_SLOTS];
                for op in body {
                    match op {
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result,
                            binding,
                            ..
                        }) => {
                            let source = bindings
                                .iter()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuI32MapReferenceError::MissingBinding(*binding))?;
                            let value = match &source.slice {
                                PcuHostScalarSlice::Read(slice) => slice[logical],
                                PcuHostScalarSlice::ReadWrite(slice) => slice[logical],
                            };
                            values[usize::from(result.0)] = value;
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            result,
                            op,
                            lhs,
                            rhs,
                            ..
                        }) => {
                            let a = values[usize::from(lhs.0)];
                            let b = values[usize::from(rhs.0)];
                            values[usize::from(result.0)] = match op {
                                PcuDispatchAluOp::Add => a.wrapping_add(b),
                                PcuDispatchAluOp::Sub => a.wrapping_sub(b),
                                PcuDispatchAluOp::Mul => a.wrapping_mul(b),
                                _ => unreachable!("validator admits only wrapping Add/Sub/Mul"),
                            };
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding,
                            value,
                            ..
                        }) => {
                            let destination = bindings
                                .iter_mut()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuI32MapReferenceError::MissingBinding(*binding))?;
                            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice
                            else {
                                return Err(PcuI32MapReferenceError::AccessMismatch(*binding));
                            };
                            slice[logical] = values[usize::from(value.0)];
                        }
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {}
                        _ => unreachable!("validator admits only i32 map operations"),
                    }
                }
                if !grid || extent - logical <= stride {
                    break;
                }
                logical += stride;
            }
        }
        Ok(())
    }
}

fn run_checked_i32_div_rem(
    submission: PcuDispatchSubmission<'_>,
    bindings: &mut [PcuHostScalarBinding<'_, i32>],
) -> Result<(), PcuI32MapReferenceError> {
    let extent = match submission.kernel.ops {
        [PcuDispatchOp::GridStrideLoop { extent, .. }, _] => *extent as usize,
        _ => submission.shape.invocation_count().get() as usize,
    };
    let [input_a, input_b, quotient, remainder] = [
        submission.kernel.bindings[0].reference(),
        submission.kernel.bindings[1].reference(),
        submission.kernel.bindings[2].reference(),
        submission.kernel.bindings[3].reference(),
    ];
    // Scan in logical order before writing so every fault leaves both outputs untouched.
    for logical in 0..extent {
        let lhs = read_i32_binding(bindings, input_a, logical)
            .ok_or(PcuI32MapReferenceError::MissingBinding(input_a))?;
        let rhs = read_i32_binding(bindings, input_b, logical)
            .ok_or(PcuI32MapReferenceError::MissingBinding(input_b))?;
        let kind = if rhs == 0 {
            Some(PcuExecutionFaultKind::DivideByZero)
        } else if lhs == i32::MIN && rhs == -1 {
            Some(PcuExecutionFaultKind::SignedDivisionOverflow)
        } else {
            None
        };
        if let Some(kind) = kind {
            return Err(PcuI32MapReferenceError::Fault(PcuExecutionFault {
                kind,
                invocation_id: u64::try_from(logical).unwrap_or(u64::MAX),
            }));
        }
    }
    for logical in 0..extent {
        let lhs = read_i32_binding(bindings, input_a, logical)
            .ok_or(PcuI32MapReferenceError::MissingBinding(input_a))?;
        let rhs = read_i32_binding(bindings, input_b, logical)
            .ok_or(PcuI32MapReferenceError::MissingBinding(input_b))?;
        for (target, value) in [(quotient, lhs / rhs), (remainder, lhs % rhs)] {
            let destination = bindings
                .iter_mut()
                .find(|candidate| candidate.target == target)
                .ok_or(PcuI32MapReferenceError::MissingBinding(target))?;
            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice else {
                return Err(PcuI32MapReferenceError::AccessMismatch(target));
            };
            let Some(output) = slice.get_mut(logical) else {
                return Err(PcuI32MapReferenceError::InvalidSubmission);
            };
            *output = value;
        }
    }
    Ok(())
}

fn read_i32_binding(
    bindings: &[PcuHostScalarBinding<'_, i32>],
    target: PcuBindingRef,
    logical: usize,
) -> Option<i32> {
    bindings
        .iter()
        .find(|candidate| candidate.target == target)
        .and_then(|candidate| match &candidate.slice {
            PcuHostScalarSlice::Read(slice) => slice.get(logical).copied(),
            PcuHostScalarSlice::ReadWrite(slice) => slice.get(logical).copied(),
        })
}

/// Failure to execute the typed wrapping `i8` map reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuI8MapReferenceError {
    InvalidSubmission,
    InvalidKernel(fusion_pcu::PcuI8MapValidationError),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
}

/// CPU oracle for the bounded typed `i8` indexed arithmetic map.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuI8MapReference;

// SAFETY: Executes synchronously on caller-owned slices and retains no references after return.
unsafe impl PcuSynchronousHostDispatchBackend<i8> for PcuI8MapReference {
    type Error = PcuI8MapReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, i8>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<i8, ()>(submission, bindings)
            .map_err(|_| PcuI8MapReferenceError::InvalidSubmission)?;
        validate_i8_map_kernel(submission.kernel).map_err(PcuI8MapReferenceError::InvalidKernel)?;
        let (body, extent, grid) = match submission.kernel.ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (*body, *extent as usize, true),
            ops => (
                ops,
                submission.shape.invocation_count().get() as usize,
                false,
            ),
        };
        let stride = submission.shape.invocation_count().get() as usize;
        for invocation in 0..stride {
            let mut logical = invocation;
            while logical < extent {
                let mut values = [0_i8; VALUE_SLOTS];
                for op in body {
                    match op {
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result,
                            binding,
                            ..
                        }) => {
                            let source = bindings
                                .iter()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuI8MapReferenceError::MissingBinding(*binding))?;
                            let value = match &source.slice {
                                PcuHostScalarSlice::Read(slice) => slice[logical],
                                PcuHostScalarSlice::ReadWrite(slice) => slice[logical],
                            };
                            values[usize::from(result.0)] = value;
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            result,
                            op,
                            lhs,
                            rhs,
                            ..
                        }) => {
                            let a = values[usize::from(lhs.0)];
                            let b = values[usize::from(rhs.0)];
                            values[usize::from(result.0)] = match op {
                                PcuDispatchAluOp::Add => a.wrapping_add(b),
                                PcuDispatchAluOp::Sub => a.wrapping_sub(b),
                                PcuDispatchAluOp::Mul => a.wrapping_mul(b),
                                _ => unreachable!("validator admits only wrapping Add/Sub/Mul"),
                            };
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding,
                            value,
                            ..
                        }) => {
                            let destination = bindings
                                .iter_mut()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuI8MapReferenceError::MissingBinding(*binding))?;
                            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice
                            else {
                                return Err(PcuI8MapReferenceError::AccessMismatch(*binding));
                            };
                            slice[logical] = values[usize::from(value.0)];
                        }
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {}
                        _ => unreachable!("validator admits only i8 map operations"),
                    }
                }
                if !grid || extent - logical <= stride {
                    break;
                }
                logical += stride;
            }
        }
        Ok(())
    }
}

/// Failure to execute the typed wrapping `i16` map reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuI16MapReferenceError {
    InvalidSubmission,
    InvalidKernel(fusion_pcu::PcuI16MapValidationError),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
}

/// CPU oracle for the bounded typed `i16` indexed arithmetic map.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuI16MapReference;

// SAFETY: Executes synchronously on caller-owned slices and retains no references after return.
unsafe impl PcuSynchronousHostDispatchBackend<i16> for PcuI16MapReference {
    type Error = PcuI16MapReferenceError;

    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, i16>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        validate_host_scalar_bindings::<i16, ()>(submission, bindings)
            .map_err(|_| PcuI16MapReferenceError::InvalidSubmission)?;
        validate_i16_map_kernel(submission.kernel)
            .map_err(PcuI16MapReferenceError::InvalidKernel)?;
        let (body, extent, grid) = match submission.kernel.ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (*body, *extent as usize, true),
            ops => (
                ops,
                submission.shape.invocation_count().get() as usize,
                false,
            ),
        };
        let stride = submission.shape.invocation_count().get() as usize;
        for invocation in 0..stride {
            let mut logical = invocation;
            while logical < extent {
                let mut values = [0_i16; VALUE_SLOTS];
                for op in body {
                    match op {
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result,
                            binding,
                            ..
                        }) => {
                            let source = bindings
                                .iter()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuI16MapReferenceError::MissingBinding(*binding))?;
                            let value = match &source.slice {
                                PcuHostScalarSlice::Read(slice) => slice[logical],
                                PcuHostScalarSlice::ReadWrite(slice) => slice[logical],
                            };
                            values[usize::from(result.0)] = value;
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            result,
                            op,
                            lhs,
                            rhs,
                            ..
                        }) => {
                            let a = values[usize::from(lhs.0)];
                            let b = values[usize::from(rhs.0)];
                            values[usize::from(result.0)] = match op {
                                PcuDispatchAluOp::Add => a.wrapping_add(b),
                                PcuDispatchAluOp::Sub => a.wrapping_sub(b),
                                PcuDispatchAluOp::Mul => a.wrapping_mul(b),
                                _ => unreachable!("validator admits only wrapping Add/Sub/Mul"),
                            };
                        }
                        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding,
                            value,
                            ..
                        }) => {
                            let destination = bindings
                                .iter_mut()
                                .find(|candidate| candidate.target == *binding)
                                .ok_or(PcuI16MapReferenceError::MissingBinding(*binding))?;
                            let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice
                            else {
                                return Err(PcuI16MapReferenceError::AccessMismatch(*binding));
                            };
                            slice[logical] = values[usize::from(value.0)];
                        }
                        PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {}
                        _ => unreachable!("validator admits only i16 map operations"),
                    }
                }
                if !grid || extent - logical <= stride {
                    break;
                }
                logical += stride;
            }
        }
        Ok(())
    }
}

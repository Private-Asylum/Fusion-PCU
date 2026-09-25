//! Opt-in CPU execution oracle for the current scalar `f32` Dispatch subset.
//!
//! This reference deliberately covers only the direct indexed-map profile shared with the
//! current `ROCm` and SPIR-V lowerers. It validates the whole program before writing outputs.

#![no_std]

use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuParameterValue,
    PcuSynchronousHostDispatchBackend,
    PcuStreamKernelIr,
    PcuStreamPattern,
    PcuStreamValueType,
    validate_host_scalar_bindings,
};

/// Failure to interpret the normative one-pattern U32 stream profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuU32StreamReferenceError {
    InvalidPortShape,
    KernelBindingsPresent,
    ParametersPresent,
    InvalidPatternCount(usize),
    UnsupportedPattern(PcuStreamPattern),
    InvalidPattern(PcuStreamPattern),
}

/// Stateless, one-word CPU reference for the common U32 Stream transform profile.
///
/// A call consumes one logical input word and returns its corresponding output word. FIFO
/// framing, buffering, and end-of-stream behavior remain the responsibility of an adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuU32StreamReference;

impl PcuU32StreamReference {
    /// Validates and applies the kernel's single U32 transform to one word.
    ///
    /// # Errors
    ///
    /// Returns the first profile shape or pattern error.
    pub fn transform(
        &self,
        kernel: &PcuStreamKernelIr<'_>,
        input: u32,
    ) -> Result<u32, PcuU32StreamReferenceError> {
        if kernel.simple_transform_type() != Some(PcuStreamValueType::U32) {
            return Err(PcuU32StreamReferenceError::InvalidPortShape);
        }
        if !kernel.bindings.is_empty() {
            return Err(PcuU32StreamReferenceError::KernelBindingsPresent);
        }
        if !kernel.parameters.is_empty() {
            return Err(PcuU32StreamReferenceError::ParametersPresent);
        }
        let [pattern] = kernel.patterns else {
            return Err(PcuU32StreamReferenceError::InvalidPatternCount(
                kernel.patterns.len(),
            ));
        };
        apply_u32_stream_pattern(*pattern, input)
    }
}

fn apply_u32_stream_pattern(
    pattern: PcuStreamPattern,
    input: u32,
) -> Result<u32, PcuU32StreamReferenceError> {
    let value = match pattern {
        PcuStreamPattern::BitReverse => input.reverse_bits(),
        PcuStreamPattern::BitInvert => !input,
        PcuStreamPattern::Increment => input.wrapping_add(1),
        PcuStreamPattern::Decrement => input.wrapping_sub(1),
        PcuStreamPattern::ShiftLeft { bits } if (1..=32).contains(&bits) => {
            if bits == 32 {
                0
            } else {
                input << bits
            }
        }
        PcuStreamPattern::ShiftRight { bits } if (1..=32).contains(&bits) => {
            if bits == 32 {
                0
            } else {
                input >> bits
            }
        }
        PcuStreamPattern::ExtractBits { offset, width }
            if width >= 1 && offset < 32 && u16::from(offset) + u16::from(width) <= 32 =>
        {
            let mask = if width == 32 {
                u32::MAX
            } else {
                (1_u32 << width) - 1
            };
            (input >> offset) & mask
        }
        PcuStreamPattern::MaskLower { bits } if (1..=32).contains(&bits) => {
            if bits == 32 {
                input
            } else {
                input & ((1_u32 << bits) - 1)
            }
        }
        PcuStreamPattern::ByteSwap32 => input.swap_bytes(),
        PcuStreamPattern::AddParameter { .. } | PcuStreamPattern::XorParameter { .. } => {
            return Err(PcuU32StreamReferenceError::UnsupportedPattern(pattern));
        }
        _ => return Err(PcuU32StreamReferenceError::InvalidPattern(pattern)),
    };
    Ok(value)
}

const VALUE_SLOTS: usize = 256;

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
        validate_program(submission, bindings)?;
        for invocation in 0..submission.shape.invocation_count().get() as usize {
            let mut values = [None; VALUE_SLOTS];
            for op in submission.kernel.ops {
                match op {
                    PcuDispatchOp::GridStrideLoop { extent, body } => {
                        let stride = submission.shape.invocation_count().get() as usize;
                        let mut logical = invocation;
                        while logical < *extent as usize {
                            execute_grid_stride_body(body, logical, bindings)?;
                            if (*extent as usize - logical) <= stride {
                                break;
                            }
                            logical += stride;
                        }
                    }
                    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                        result, binding, ..
                    }) => {
                        let source = bindings
                            .iter()
                            .find(|candidate| candidate.target == *binding)
                            .ok_or(PcuF32ReferenceError::MissingBinding(*binding))?;
                        let value = match &source.slice {
                            PcuHostScalarSlice::Read(slice) => slice[invocation],
                            PcuHostScalarSlice::ReadWrite(slice) => slice[invocation],
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
                            PcuDispatchAluOp::Min => f32_min(left, right),
                            PcuDispatchAluOp::Max => f32_max(left, right),
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

fn validate_program(
    submission: PcuDispatchSubmission<'_>,
    bindings: &[PcuHostScalarBinding<'_, f32>],
) -> Result<(), PcuF32ReferenceError> {
    if let Some((position, extent, body)) =
        submission
            .kernel
            .ops
            .iter()
            .enumerate()
            .find_map(|(position, op)| match op {
                PcuDispatchOp::GridStrideLoop { extent, body } => Some((position, extent, body)),
                _ => None,
            })
    {
        if *extent == 0 || submission.kernel.ops.len() != 2 || position != 0 {
            return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
        }
        if !matches!(
            submission.kernel.ops[1],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return)
        ) {
            return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
        }
        validate_grid_stride_body(submission.kernel, body, bindings)?;
        return Ok(());
    }
    let mut defined = [false; VALUE_SLOTS];
    let mut saw_store = false;
    let mut saw_return = false;
    for (position, op) in submission.kernel.ops.iter().copied().enumerate() {
        if saw_return {
            return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
        }
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                check_index(index, position)?;
                check_binding(submission.kernel, bindings, binding, false)?;
                define(&mut defined, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
                }
                define(&mut defined, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
            }) => {
                if !matches!(
                    op,
                    PcuDispatchAluOp::Add
                        | PcuDispatchAluOp::Sub
                        | PcuDispatchAluOp::Mul
                        | PcuDispatchAluOp::Div
                        | PcuDispatchAluOp::Min
                        | PcuDispatchAluOp::Max
                ) {
                    return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
                }
                require(&defined, lhs)?;
                require(&defined, rhs)?;
                define(&mut defined, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index,
                value,
            }) => {
                check_index(index, position)?;
                check_binding(submission.kernel, bindings, binding, true)?;
                require(&defined, value)?;
                saw_store = true;
            }
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) => saw_return = true,
            _ => return Err(PcuF32ReferenceError::UnsupportedInstruction(position)),
        }
    }
    if !saw_store {
        return Err(PcuF32ReferenceError::MissingStore);
    }
    if !saw_return {
        return Err(PcuF32ReferenceError::MissingReturn);
    }
    Ok(())
}

fn f32_min(left: f32, right: f32) -> f32 {
    if left == 0.0 && right == 0.0 {
        return f32::from_bits(1 << 31);
    }
    left.min(right)
}

fn f32_max(left: f32, right: f32) -> f32 {
    if left == 0.0 && right == 0.0 {
        return 0.0;
    }
    left.max(right)
}

fn validate_grid_stride_body(
    kernel: &PcuDispatchKernelIr<'_>,
    body: &[PcuDispatchOp<'_>],
    bindings: &[PcuHostScalarBinding<'_, f32>],
) -> Result<(), PcuF32ReferenceError> {
    let mut saw_store = false;
    for (position, op) in body.iter().copied().enumerate() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                check_grid_index(index, position)?;
                check_binding(kernel, bindings, binding, false)?;
                grid_define(body, position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
                }
                grid_define(body, position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
            }) => {
                if !matches!(
                    op,
                    PcuDispatchAluOp::Add
                        | PcuDispatchAluOp::Sub
                        | PcuDispatchAluOp::Mul
                        | PcuDispatchAluOp::Div
                        | PcuDispatchAluOp::Min
                        | PcuDispatchAluOp::Max
                ) {
                    return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
                }
                grid_require(body, position, lhs)?;
                grid_require(body, position, rhs)?;
                grid_define(body, position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index,
                value,
            }) => {
                check_grid_index(index, position)?;
                check_binding(kernel, bindings, binding, true)?;
                grid_require(body, position, value)?;
                saw_store = true;
            }
            _ => return Err(PcuF32ReferenceError::UnsupportedInstruction(position)),
        }
    }
    if saw_store {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::MissingStore)
    }
}

fn execute_grid_stride_body(
    body: &[PcuDispatchOp<'_>],
    logical: usize,
    bindings: &mut [PcuHostScalarBinding<'_, f32>],
) -> Result<(), PcuF32ReferenceError> {
    let mut values = [None; VALUE_SLOTS];
    for op in body {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result, binding, ..
            }) => {
                let source = bindings
                    .iter()
                    .find(|candidate| candidate.target == *binding)
                    .ok_or(PcuF32ReferenceError::MissingBinding(*binding))?;
                let value = match &source.slice {
                    PcuHostScalarSlice::Read(slice) => slice[logical],
                    PcuHostScalarSlice::ReadWrite(slice) => slice[logical],
                };
                values[usize::from(result.0)] = Some(value);
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result,
                value: PcuParameterValue::F32(bits),
            }) => {
                values[usize::from(result.0)] = Some(f32::from_bits(*bits));
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
            }) => {
                let left =
                    values[usize::from(lhs.0)].ok_or(PcuF32ReferenceError::InvalidValue(*lhs))?;
                let right =
                    values[usize::from(rhs.0)].ok_or(PcuF32ReferenceError::InvalidValue(*rhs))?;
                values[usize::from(result.0)] = Some(match op {
                    PcuDispatchAluOp::Add => left + right,
                    PcuDispatchAluOp::Sub => left - right,
                    PcuDispatchAluOp::Mul => left * right,
                    PcuDispatchAluOp::Div => left / right,
                    PcuDispatchAluOp::Min => f32_min(left, right),
                    PcuDispatchAluOp::Max => f32_max(left, right),
                    _ => return Err(PcuF32ReferenceError::UnsupportedInstruction(0)),
                });
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
                let destination = bindings
                    .iter_mut()
                    .find(|candidate| candidate.target == *binding)
                    .ok_or(PcuF32ReferenceError::MissingBinding(*binding))?;
                let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice else {
                    return Err(PcuF32ReferenceError::AccessMismatch(*binding));
                };
                slice[logical] = values[usize::from(value.0)]
                    .ok_or(PcuF32ReferenceError::InvalidValue(*value))?;
            }
            _ => return Err(PcuF32ReferenceError::UnsupportedInstruction(0)),
        }
    }
    Ok(())
}

const fn check_grid_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF32ReferenceError> {
    if matches!(index, PcuDispatchIndex::GridStrideId) {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::UnsupportedInstruction(position))
    }
}

fn grid_define(
    body: &[PcuDispatchOp<'_>],
    position: usize,
    id: PcuDispatchValueId,
) -> Result<(), PcuF32ReferenceError> {
    if usize::from(id.0) >= VALUE_SLOTS
        || id.0 == 0
        || body[..position].iter().any(|op| match op {
            PcuDispatchOp::Data(
                PcuDispatchDataOp::BindingLoad { result, .. }
                | PcuDispatchDataOp::Constant { result, .. }
                | PcuDispatchDataOp::Alu { result, .. },
            ) => *result == id,
            _ => false,
        })
    {
        Err(PcuF32ReferenceError::InvalidValue(id))
    } else {
        Ok(())
    }
}

fn grid_require(
    body: &[PcuDispatchOp<'_>],
    position: usize,
    id: PcuDispatchValueId,
) -> Result<(), PcuF32ReferenceError> {
    if usize::from(id.0) < VALUE_SLOTS
        && body[..position].iter().any(|op| match op {
            PcuDispatchOp::Data(
                PcuDispatchDataOp::BindingLoad { result, .. }
                | PcuDispatchDataOp::Constant { result, .. }
                | PcuDispatchDataOp::Alu { result, .. },
            ) => *result == id,
            _ => false,
        })
    {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::InvalidValue(id))
    }
}

const fn check_index(index: PcuDispatchIndex, position: usize) -> Result<(), PcuF32ReferenceError> {
    if matches!(index, PcuDispatchIndex::InvocationId) {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::UnsupportedInstruction(position))
    }
}

fn check_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    bindings: &[PcuHostScalarBinding<'_, f32>],
    target: PcuBindingRef,
    write: bool,
) -> Result<(), PcuF32ReferenceError> {
    let Some(declared) = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == target)
    else {
        return Err(PcuF32ReferenceError::MissingBinding(target));
    };
    let Some(binding) = bindings.iter().find(|binding| binding.target == target) else {
        return Err(PcuF32ReferenceError::MissingBinding(target));
    };
    if (write
        && (declared.access == PcuBindingAccess::ReadOnly
            || binding.slice.access() != PcuBindingAccess::ReadWrite))
        || (!write && declared.access == PcuBindingAccess::WriteOnly)
    {
        return Err(PcuF32ReferenceError::AccessMismatch(target));
    }
    Ok(())
}

fn define(
    defined: &mut [bool; VALUE_SLOTS],
    id: PcuDispatchValueId,
) -> Result<(), PcuF32ReferenceError> {
    let Some(slot) = defined.get_mut(usize::from(id.0)) else {
        return Err(PcuF32ReferenceError::InvalidValue(id));
    };
    if id.0 == 0 || *slot {
        return Err(PcuF32ReferenceError::InvalidValue(id));
    }
    *slot = true;
    Ok(())
}

fn require(
    defined: &[bool; VALUE_SLOTS],
    id: PcuDispatchValueId,
) -> Result<(), PcuF32ReferenceError> {
    if defined.get(usize::from(id.0)) == Some(&true) {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::InvalidValue(id))
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use super::{
        PcuF32Reference,
        PcuF32ReferenceError,
        PcuU32StreamReference,
        PcuU32StreamReferenceError,
    };
    use fusion_pcu::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchAluOp,
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchSubmission,
        PcuDispatchValueId,
        PcuDispatchFeatureCaps,
        PcuHostScalarBinding,
        PcuHostScalarSlice,
        PcuInvocationParameters,
        PcuInvocationShape,
        PcuKernelId,
        PcuParameterValue,
        PcuScalarType,
        PcuSynchronousHostDispatchBackend,
        PcuValueType,
        PcuValueTypeCaps,
        PcuStreamPattern,
    };
    use fusion_pcu::model::PcuStreamKernelBuilder;

    #[test]
    fn u32_stream_reference_matches_common_profile_vectors() {
        let vectors = [
            (PcuStreamPattern::BitReverse, 0x0000_0001, 0x8000_0000),
            (PcuStreamPattern::BitInvert, 0x00ff_00ff, 0xff00_ff00),
            (PcuStreamPattern::Increment, u32::MAX, 0),
            (PcuStreamPattern::Decrement, 0, u32::MAX),
            (
                PcuStreamPattern::ShiftLeft { bits: 3 },
                0x8000_0003,
                0x0000_0018,
            ),
            (
                PcuStreamPattern::ShiftRight { bits: 4 },
                0x8000_003f,
                0x0800_0003,
            ),
            (PcuStreamPattern::ShiftLeft { bits: 32 }, 0x1234_5678, 0),
            (
                PcuStreamPattern::ExtractBits {
                    offset: 8,
                    width: 8,
                },
                0x1234_56ab,
                0x56,
            ),
            (PcuStreamPattern::MaskLower { bits: 12 }, 0xabcd_1234, 0x234),
            (PcuStreamPattern::ByteSwap32, 0x1234_56ab, 0xab56_3412),
        ];
        let reference = PcuU32StreamReference;
        for (pattern, input, expected) in vectors {
            let builder = PcuStreamKernelBuilder::<1>::words(7, "cpu_reference")
                .with_pattern(pattern)
                .expect("one pattern fits");
            assert_eq!(reference.transform(&builder.ir(), input), Ok(expected));
        }
    }

    #[test]
    fn u32_stream_reference_rejects_non_profile_patterns_and_composition() {
        let parameterized = PcuStreamKernelBuilder::<1>::words(8, "parameterized")
            .with_pattern(PcuStreamPattern::AddParameter {
                parameter: fusion_pcu::PcuParameterSlot(0),
            })
            .expect("one pattern fits");
        assert_eq!(
            PcuU32StreamReference.transform(&parameterized.ir(), 1),
            Err(PcuU32StreamReferenceError::UnsupportedPattern(
                PcuStreamPattern::AddParameter {
                    parameter: fusion_pcu::PcuParameterSlot(0),
                }
            ))
        );

        let composed = PcuStreamKernelBuilder::<2>::words(9, "composed")
            .increment()
            .expect("pattern fits")
            .decrement()
            .expect("pattern fits");
        assert_eq!(
            PcuU32StreamReference.transform(&composed.ir(), 1),
            Err(PcuU32StreamReferenceError::InvalidPatternCount(2))
        );

        let invalid_shift = PcuStreamKernelBuilder::<1>::words(10, "invalid_shift")
            .with_pattern(PcuStreamPattern::ShiftRight { bits: 0 })
            .expect("builder preserves the invalid pattern for backend validation");
        assert_eq!(
            PcuU32StreamReference.transform(&invalid_shift.ir(), 1),
            Err(PcuU32StreamReferenceError::InvalidPattern(
                PcuStreamPattern::ShiftRight { bits: 0 }
            ))
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn min_max_execute_and_unsupported_programs_preserve_outputs() {
        let bindings_ir = [
            PcuBinding::value(
                Some("lhs"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("rhs"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("min"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("max"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let id0 = PcuDispatchValueId(1);
        let id1 = PcuDispatchValueId(2);
        let id2 = PcuDispatchValueId(3);
        let id3 = PcuDispatchValueId(4);
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: id0,
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: id1,
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result: id2,
                op: PcuDispatchAluOp::Min,
                lhs: id0,
                rhs: id1,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result: id3,
                op: PcuDispatchAluOp::Max,
                lhs: id0,
                rhs: id1,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: id2,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::InvocationId,
                value: id3,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "min_max",
                logical_shape: [8, 1, 1],
            },
            bindings: &bindings_ir,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::F32),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(8).expect("nonzero")),
        };
        let lhs = [1.0, 9.0, -2.0, -0.0, f32::NAN, f32::NAN, 0.0, 2.0];
        let rhs = [3.0, 4.0, -5.0, 0.0, 2.0, f32::NAN, -0.0, f32::NAN];
        let mut minimum = [0.0; 8];
        let mut maximum = [0.0; 8];
        let mut host_bindings = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&lhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&rhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut minimum),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut maximum),
            },
        ];
        PcuF32Reference
            .run_host_direct(
                submission,
                &mut host_bindings,
                PcuInvocationParameters::empty(),
            )
            .expect("min and max execute");
        let expected_minimum = [1.0_f32, 4.0, -5.0, -0.0, 2.0, 0.0, -0.0, 2.0];
        let expected_maximum = [3.0_f32, 9.0, -2.0, 0.0, 2.0, 0.0, 0.0, 2.0];
        for index in [0, 1, 2, 3, 4, 6, 7] {
            assert_eq!(minimum[index].to_bits(), expected_minimum[index].to_bits());
            assert_eq!(maximum[index].to_bits(), expected_maximum[index].to_bits());
        }
        assert!(minimum[5].is_nan());
        assert!(maximum[5].is_nan());
        assert!(minimum[6].is_sign_negative());
        assert!(maximum[6].is_sign_positive());

        let unsupported_ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: id0,
                value: PcuParameterValue::F32(0),
            }),
            PcuDispatchOp::Arithmetic(PcuDispatchAluOp::And),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: id0,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let unsupported_kernel = PcuDispatchKernelIr {
            ops: &unsupported_ops,
            ..kernel
        };
        let unsupported_submission = PcuDispatchSubmission {
            kernel: &unsupported_kernel,
            ..submission
        };
        let mut untouched = [7.0; 8];
        let mut unsupported_bindings = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&lhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&rhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut untouched),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut maximum),
            },
        ];
        assert_eq!(
            PcuF32Reference.run_host_direct(
                unsupported_submission,
                &mut unsupported_bindings,
                PcuInvocationParameters::empty()
            ),
            Err(PcuF32ReferenceError::UnsupportedInstruction(1))
        );
        assert_eq!(untouched.map(f32::to_bits), [7.0_f32; 8].map(f32::to_bits));
    }

    #[test]
    fn grid_stride_reference_processes_extent_larger_than_launch_width() {
        use fusion_pcu::validate_f32_map_kernel;
        let bindings = [
            PcuBinding::value(
                Some("source"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("destination"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(2),
                value: PcuParameterValue::F32(1.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
        ];
        let ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 7,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(18),
            entry: PcuDispatchEntryPoint {
                name: "grid_stride",
                logical_shape: [2, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::F32),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        assert_eq!(validate_f32_map_kernel(&kernel), Ok(()));
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
        };
        // Exact normal values keep this vector inside the portable cross-backend numeric domain.
        let source = [1.25, 2.5, 3.75, 4.0, 5.0, 6.0, 7.0];
        let mut destination = [0.0; 7];
        let mut host_bindings = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&source),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::ReadWrite(&mut destination),
            },
        ];
        PcuF32Reference
            .run_host_direct(
                submission,
                &mut host_bindings,
                PcuInvocationParameters::empty(),
            )
            .expect("grid stride covers the logical extent");
        assert_eq!(
            destination.map(f32::to_bits),
            [2.25_f32, 3.5, 4.75, 5.0, 6.0, 7.0, 8.0].map(f32::to_bits)
        );
    }
}

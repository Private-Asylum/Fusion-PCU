//! Reference execution for bounded heterogeneous scalar conversion profiles.

#[rustfmt::skip]
use fusion_pcu::{
    validate_dispatch_submission,
    validate_typed_dispatch_value_flow,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchConversion,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuF16Bits,
    PcuBf16Bits,
    f32_bits_to_f64_bits,
    PcuScalarType,
    PcuTypedDispatchValidationError,
    PcuValueType,
};

const VALUE_SLOTS: usize = 256;

/// One heterogeneous host resource for the CPU typed-conversion reference profile.
#[derive(Debug)]
pub struct PcuCpuTypedBinding<'a> {
    pub target: PcuBindingRef,
    pub slice: PcuCpuTypedSlice<'a>,
}

/// Borrowed storage supported by the CPU heterogeneous conversion reference.
#[derive(Debug)]
pub enum PcuCpuTypedSlice<'a> {
    ReadI8(&'a [i8]),
    ReadU8(&'a [u8]),
    ReadWriteI8(&'a mut [i8]),
    ReadWriteU8(&'a mut [u8]),
    ReadI16(&'a [i16]),
    ReadU16(&'a [u16]),
    ReadWriteI16(&'a mut [i16]),
    ReadWriteU16(&'a mut [u16]),
    ReadI32(&'a [i32]),
    ReadU32(&'a [u32]),
    ReadWriteI32(&'a mut [i32]),
    ReadWriteU32(&'a mut [u32]),
    ReadI64(&'a [i64]),
    ReadU64(&'a [u64]),
    ReadWriteI64(&'a mut [i64]),
    ReadWriteU64(&'a mut [u64]),
    ReadF32(&'a [f32]),
    ReadWriteF32(&'a mut [f32]),
    ReadF64(&'a [f64]),
    ReadWriteF64(&'a mut [f64]),
    ReadF16Bits(&'a [PcuF16Bits]),
    ReadWriteF16Bits(&'a mut [PcuF16Bits]),
    ReadBf16Bits(&'a [PcuBf16Bits]),
    ReadWriteBf16Bits(&'a mut [PcuBf16Bits]),
}

impl PcuCpuTypedSlice<'_> {
    const fn value_type(&self) -> PcuValueType {
        match self {
            Self::ReadI8(_) | Self::ReadWriteI8(_) => PcuValueType::Scalar(PcuScalarType::I8),
            Self::ReadU8(_) | Self::ReadWriteU8(_) => PcuValueType::Scalar(PcuScalarType::U8),
            Self::ReadI16(_) | Self::ReadWriteI16(_) => PcuValueType::Scalar(PcuScalarType::I16),
            Self::ReadU16(_) | Self::ReadWriteU16(_) => PcuValueType::Scalar(PcuScalarType::U16),
            Self::ReadI32(_) | Self::ReadWriteI32(_) => PcuValueType::Scalar(PcuScalarType::I32),
            Self::ReadU32(_) | Self::ReadWriteU32(_) => PcuValueType::Scalar(PcuScalarType::U32),
            Self::ReadI64(_) | Self::ReadWriteI64(_) => PcuValueType::Scalar(PcuScalarType::I64),
            Self::ReadU64(_) | Self::ReadWriteU64(_) => PcuValueType::Scalar(PcuScalarType::U64),
            Self::ReadF32(_) | Self::ReadWriteF32(_) => PcuValueType::Scalar(PcuScalarType::F32),
            Self::ReadF64(_) | Self::ReadWriteF64(_) => PcuValueType::Scalar(PcuScalarType::F64),
            Self::ReadF16Bits(_) | Self::ReadWriteF16Bits(_) => {
                PcuValueType::Scalar(PcuScalarType::F16)
            }
            Self::ReadBf16Bits(_) | Self::ReadWriteBf16Bits(_) => {
                PcuValueType::Scalar(PcuScalarType::BF16)
            }
        }
    }

    const fn len(&self) -> usize {
        match self {
            Self::ReadI8(slice) => slice.len(),
            Self::ReadU8(slice) => slice.len(),
            Self::ReadWriteI8(slice) => slice.len(),
            Self::ReadWriteU8(slice) => slice.len(),
            Self::ReadI16(slice) => slice.len(),
            Self::ReadU16(slice) => slice.len(),
            Self::ReadWriteI16(slice) => slice.len(),
            Self::ReadWriteU16(slice) => slice.len(),
            Self::ReadI32(slice) => slice.len(),
            Self::ReadU32(slice) => slice.len(),
            Self::ReadWriteI32(slice) => slice.len(),
            Self::ReadWriteU32(slice) => slice.len(),
            Self::ReadI64(slice) => slice.len(),
            Self::ReadU64(slice) => slice.len(),
            Self::ReadWriteI64(slice) => slice.len(),
            Self::ReadWriteU64(slice) => slice.len(),
            Self::ReadF32(slice) => slice.len(),
            Self::ReadWriteF32(slice) => slice.len(),
            Self::ReadF64(slice) => slice.len(),
            Self::ReadWriteF64(slice) => slice.len(),
            Self::ReadF16Bits(slice) => slice.len(),
            Self::ReadWriteF16Bits(slice) => slice.len(),
            Self::ReadBf16Bits(slice) => slice.len(),
            Self::ReadWriteBf16Bits(slice) => slice.len(),
        }
    }

    const fn is_writable(&self) -> bool {
        matches!(
            self,
            Self::ReadWriteI8(_)
                | Self::ReadWriteU8(_)
                | Self::ReadWriteI16(_)
                | Self::ReadWriteU16(_)
                | Self::ReadWriteI32(_)
                | Self::ReadWriteU32(_)
                | Self::ReadWriteI64(_)
                | Self::ReadWriteU64(_)
                | Self::ReadWriteF32(_)
                | Self::ReadWriteF64(_)
                | Self::ReadWriteF16Bits(_)
                | Self::ReadWriteBf16Bits(_)
        )
    }
}

/// Rejection or execution failure for the CPU typed conversion oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuTypedConversionReferenceError {
    UnsupportedNumericalRequirements,
    InvalidSubmission,
    InvalidValueFlow(PcuTypedDispatchValidationError),
    UnsupportedProfile,
    DuplicateBinding(PcuBindingRef),
    MissingBinding(PcuBindingRef),
    UnexpectedBinding(PcuBindingRef),
    TypeMismatch(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
    UnsupportedLayout(PcuBindingRef),
    BufferTooSmall(PcuBindingRef),
    InvalidValue(PcuDispatchValueId),
}

/// Synchronous CPU oracle for integer widening and specified f32/half Dispatch conversions.
///
/// Supports one direct region or one grid-stride region. Each invocation starts with fresh SSA
/// values, conversion uses exact sign or zero extension or the f32↔half conversion law encoded by
/// [`PcuF16Bits`] and [`PcuBf16Bits`], and output is visible on return. Narrowing uses
/// round-to-nearest, ties-to-even, quiets NaNs while retaining sign and available payload bits,
/// and maps overflow to signed infinity. Widening is exact and preserves signaling NaNs.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuTypedConversionReference;

impl PcuTypedConversionReference {
    /// Runs the typed conversion profile synchronously over heterogeneous host slices.
    ///
    /// # Errors
    ///
    /// Rejects malformed submissions, unsupported operations/layouts, incompatible host
    /// bindings, and undersized buffers before running the first invocation.
    pub fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuCpuTypedBinding<'_>],
    ) -> Result<(), PcuTypedConversionReferenceError> {
        if submission
            .kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            == fusion_pcu::PcuReproducibility::PortableV1
        {
            return Err(PcuTypedConversionReferenceError::UnsupportedNumericalRequirements);
        }
        validate_dispatch_submission(submission)
            .map_err(|_| PcuTypedConversionReferenceError::InvalidSubmission)?;
        validate_typed_dispatch_value_flow(submission.kernel)
            .map_err(PcuTypedConversionReferenceError::InvalidValueFlow)?;
        if !submission.kernel.ports.is_empty() || !submission.kernel.parameters.is_empty() {
            return Err(PcuTypedConversionReferenceError::UnsupportedProfile);
        }
        let (ops, extent, grid) = profile_region(submission.kernel)?;
        validate_profile_ops(ops, grid)?;
        validate_bindings(submission, bindings, grid, extent)?;

        let stride = usize::try_from(submission.shape.invocation_count().get())
            .map_err(|_| PcuTypedConversionReferenceError::InvalidSubmission)?;
        let extent = usize::try_from(extent)
            .map_err(|_| PcuTypedConversionReferenceError::InvalidSubmission)?;
        for lane in 0..stride {
            if grid && lane >= extent {
                break;
            }
            let mut logical = lane;
            loop {
                execute_invocation(ops, bindings, logical)?;
                if !grid || extent - logical <= stride {
                    break;
                }
                logical += stride;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
#[allow(clippy::redundant_pub_crate)] // Shared only with the sibling checked CPU executor.
pub(super) enum Value {
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    F16Bits(PcuF16Bits),
    Bf16Bits(PcuBf16Bits),
}

fn profile_region<'kernel, 'ir>(
    kernel: &'kernel PcuDispatchKernelIr<'ir>,
) -> Result<(&'kernel [PcuDispatchOp<'ir>], u32, bool), PcuTypedConversionReferenceError> {
    match kernel.ops {
        [PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return)] => {
            Err(PcuTypedConversionReferenceError::UnsupportedProfile)
        }
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
        ] => Ok((body, *extent, true)),
        ops if ops
            .iter()
            .any(|op| matches!(op, PcuDispatchOp::GridStrideLoop { .. })) =>
        {
            Err(PcuTypedConversionReferenceError::UnsupportedProfile)
        }
        ops => Ok((ops, kernel.entry.logical_shape[0], false)),
    }
}

fn validate_profile_ops(
    ops: &[PcuDispatchOp<'_>],
    grid: bool,
) -> Result<(), PcuTypedConversionReferenceError> {
    let mut conversions = 0;
    let mut stores = 0;
    let mut returned = false;
    for (position, op) in ops.iter().enumerate() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { index, .. }) => {
                let allowed = if grid {
                    matches!(
                        index,
                        PcuDispatchIndex::GridStrideId | PcuDispatchIndex::BindingElementZero
                    )
                } else {
                    matches!(
                        index,
                        PcuDispatchIndex::InvocationId | PcuDispatchIndex::BindingElementZero
                    )
                };
                if !allowed || returned {
                    return Err(PcuTypedConversionReferenceError::UnsupportedProfile);
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Convert { conversion, .. }) => {
                if returned
                    || !matches!(
                        conversion,
                        PcuDispatchConversion::I8ToI16
                            | PcuDispatchConversion::U8ToU16
                            | PcuDispatchConversion::I16ToI32
                            | PcuDispatchConversion::U16ToU32
                            | PcuDispatchConversion::I32ToI64
                            | PcuDispatchConversion::U32ToU64
                            | PcuDispatchConversion::F32ToF16Bits
                            | PcuDispatchConversion::F16BitsToF32
                            | PcuDispatchConversion::F32ToBf16Bits
                            | PcuDispatchConversion::Bf16BitsToF32
                            | PcuDispatchConversion::F32ToF64Exact
                    )
                {
                    return Err(PcuTypedConversionReferenceError::UnsupportedProfile);
                }
                conversions += 1;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { index, .. }) => {
                if returned
                    || (grid && *index != PcuDispatchIndex::GridStrideId)
                    || (!grid && *index != PcuDispatchIndex::InvocationId)
                {
                    return Err(PcuTypedConversionReferenceError::UnsupportedProfile);
                }
                stores += 1;
            }
            PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return) if !grid => {
                if returned || position + 1 != ops.len() {
                    return Err(PcuTypedConversionReferenceError::UnsupportedProfile);
                }
                returned = true;
            }
            _ => return Err(PcuTypedConversionReferenceError::UnsupportedProfile),
        }
    }
    if conversions != 1 || stores == 0 || (!grid && !returned) {
        return Err(PcuTypedConversionReferenceError::UnsupportedProfile);
    }
    Ok(())
}

#[allow(clippy::redundant_pub_crate)] // Shared only with the sibling checked CPU executor.
pub(super) fn validate_bindings(
    submission: PcuDispatchSubmission<'_>,
    bindings: &[PcuCpuTypedBinding<'_>],
    _grid: bool,
    _extent: u32,
) -> Result<(), PcuTypedConversionReferenceError> {
    for (index, binding) in bindings.iter().enumerate() {
        if bindings[..index]
            .iter()
            .any(|prior| prior.target == binding.target)
        {
            return Err(PcuTypedConversionReferenceError::DuplicateBinding(
                binding.target,
            ));
        }
        let Some(declared) = submission
            .kernel
            .bindings
            .iter()
            .find(|item| item.reference() == binding.target)
        else {
            return Err(PcuTypedConversionReferenceError::UnexpectedBinding(
                binding.target,
            ));
        };
        if declared.binding_type != PcuBindingType::Value(binding.slice.value_type()) {
            return Err(PcuTypedConversionReferenceError::TypeMismatch(
                binding.target,
            ));
        }
        if declared.storage != PcuBindingStorageClass::Storage || declared.builtin.is_some() {
            return Err(PcuTypedConversionReferenceError::UnsupportedLayout(
                binding.target,
            ));
        }
        if matches!(
            declared.access,
            PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
        ) && !binding.slice.is_writable()
        {
            return Err(PcuTypedConversionReferenceError::AccessMismatch(
                binding.target,
            ));
        }
        let used_for_load = ops_load_binding(submission.kernel, binding.target);
        let used_for_store = ops_store_binding(submission.kernel, binding.target);
        if (used_for_load && declared.access == PcuBindingAccess::WriteOnly)
            || (used_for_store && declared.access == PcuBindingAccess::ReadOnly)
        {
            return Err(PcuTypedConversionReferenceError::AccessMismatch(
                binding.target,
            ));
        }
        let required = submission
            .kernel
            .minimum_binding_elements_for(binding.target, submission.shape.invocation_count().get())
            as usize;
        if binding.slice.len() < required {
            return Err(PcuTypedConversionReferenceError::BufferTooSmall(
                binding.target,
            ));
        }
    }
    for declared in submission.kernel.bindings {
        if !bindings
            .iter()
            .any(|binding| binding.target == declared.reference())
        {
            return Err(PcuTypedConversionReferenceError::MissingBinding(
                declared.reference(),
            ));
        }
    }
    Ok(())
}

fn ops_load_binding(kernel: &PcuDispatchKernelIr<'_>, target: PcuBindingRef) -> bool {
    kernel.ops.iter().any(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, .. }) => *binding == target,
        PcuDispatchOp::GridStrideLoop { body, .. } => body.iter().any(|nested| {
            matches!(nested, PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, .. }) if *binding == target)
        }),
        _ => false,
    })
}

fn ops_store_binding(kernel: &PcuDispatchKernelIr<'_>, target: PcuBindingRef) -> bool {
    kernel.ops.iter().any(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, .. }) => *binding == target,
        PcuDispatchOp::GridStrideLoop { body, .. } => body.iter().any(|nested| {
            matches!(nested, PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, .. }) if *binding == target)
        }),
        _ => false,
    })
}

fn execute_invocation(
    ops: &[PcuDispatchOp<'_>],
    bindings: &mut [PcuCpuTypedBinding<'_>],
    logical: usize,
) -> Result<(), PcuTypedConversionReferenceError> {
    let mut values = [None; VALUE_SLOTS];
    for op in ops {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                let element = if *index == PcuDispatchIndex::BindingElementZero {
                    0
                } else {
                    logical
                };
                let source = bindings
                    .iter()
                    .find(|candidate| candidate.target == *binding)
                    .ok_or(PcuTypedConversionReferenceError::MissingBinding(*binding))?;
                values[usize::from(result.0)] = Some(load_slice_value(&source.slice, element));
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                result,
                value,
                conversion,
            }) => {
                let input = values[usize::from(value.0)]
                    .ok_or(PcuTypedConversionReferenceError::InvalidValue(*value))?;
                values[usize::from(result.0)] = Some(convert_value(*conversion, input)?);
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
                let output = values[usize::from(value.0)]
                    .ok_or(PcuTypedConversionReferenceError::InvalidValue(*value))?;
                let destination = bindings
                    .iter_mut()
                    .find(|candidate| candidate.target == *binding)
                    .ok_or(PcuTypedConversionReferenceError::MissingBinding(*binding))?;
                store_slice_value(&mut destination.slice, logical, output, *binding)?;
            }
            PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return) => {}
            _ => return Err(PcuTypedConversionReferenceError::UnsupportedProfile),
        }
    }
    Ok(())
}

#[allow(clippy::redundant_pub_crate)] // Shared only with the sibling checked CPU executor.
pub(super) const fn load_slice_value(slice: &PcuCpuTypedSlice<'_>, element: usize) -> Value {
    match slice {
        PcuCpuTypedSlice::ReadI8(slice) => Value::I8(slice[element]),
        PcuCpuTypedSlice::ReadU8(slice) => Value::U8(slice[element]),
        PcuCpuTypedSlice::ReadWriteI8(slice) => Value::I8(slice[element]),
        PcuCpuTypedSlice::ReadWriteU8(slice) => Value::U8(slice[element]),
        PcuCpuTypedSlice::ReadI16(slice) => Value::I16(slice[element]),
        PcuCpuTypedSlice::ReadU16(slice) => Value::U16(slice[element]),
        PcuCpuTypedSlice::ReadWriteI16(slice) => Value::I16(slice[element]),
        PcuCpuTypedSlice::ReadWriteU16(slice) => Value::U16(slice[element]),
        PcuCpuTypedSlice::ReadI32(slice) => Value::I32(slice[element]),
        PcuCpuTypedSlice::ReadU32(slice) => Value::U32(slice[element]),
        PcuCpuTypedSlice::ReadWriteI32(slice) => Value::I32(slice[element]),
        PcuCpuTypedSlice::ReadWriteU32(slice) => Value::U32(slice[element]),
        PcuCpuTypedSlice::ReadI64(slice) => Value::I64(slice[element]),
        PcuCpuTypedSlice::ReadU64(slice) => Value::U64(slice[element]),
        PcuCpuTypedSlice::ReadWriteI64(slice) => Value::I64(slice[element]),
        PcuCpuTypedSlice::ReadWriteU64(slice) => Value::U64(slice[element]),
        PcuCpuTypedSlice::ReadF32(slice) => Value::F32(slice[element]),
        PcuCpuTypedSlice::ReadWriteF32(slice) => Value::F32(slice[element]),
        PcuCpuTypedSlice::ReadF64(slice) => Value::F64(slice[element]),
        PcuCpuTypedSlice::ReadWriteF64(slice) => Value::F64(slice[element]),
        PcuCpuTypedSlice::ReadF16Bits(slice) => Value::F16Bits(slice[element]),
        PcuCpuTypedSlice::ReadWriteF16Bits(slice) => Value::F16Bits(slice[element]),
        PcuCpuTypedSlice::ReadBf16Bits(slice) => Value::Bf16Bits(slice[element]),
        PcuCpuTypedSlice::ReadWriteBf16Bits(slice) => Value::Bf16Bits(slice[element]),
    }
}

fn convert_value(
    conversion: PcuDispatchConversion,
    input: Value,
) -> Result<Value, PcuTypedConversionReferenceError> {
    match (conversion, input) {
        (PcuDispatchConversion::I8ToI16, Value::I8(value)) => Ok(Value::I16(i16::from(value))),
        (PcuDispatchConversion::U8ToU16, Value::U8(value)) => Ok(Value::U16(u16::from(value))),
        (PcuDispatchConversion::I16ToI32, Value::I16(value)) => Ok(Value::I32(i32::from(value))),
        (PcuDispatchConversion::U16ToU32, Value::U16(value)) => Ok(Value::U32(u32::from(value))),
        (PcuDispatchConversion::I32ToI64, Value::I32(value)) => Ok(Value::I64(i64::from(value))),
        (PcuDispatchConversion::U32ToU64, Value::U32(value)) => Ok(Value::U64(u64::from(value))),
        (PcuDispatchConversion::F32ToF16Bits, Value::F32(value)) => {
            Ok(Value::F16Bits(PcuF16Bits::from_f32(value)))
        }
        (PcuDispatchConversion::F16BitsToF32, Value::F16Bits(value)) => {
            Ok(Value::F32(value.to_f32()))
        }
        (PcuDispatchConversion::F32ToBf16Bits, Value::F32(value)) => {
            Ok(Value::Bf16Bits(PcuBf16Bits::from_f32(value)))
        }
        (PcuDispatchConversion::F32ToF64Exact, Value::F32(value)) => Ok(Value::F64(
            f64::from_bits(f32_bits_to_f64_bits(value.to_bits())),
        )),
        (PcuDispatchConversion::Bf16BitsToF32, Value::Bf16Bits(value)) => {
            Ok(Value::F32(value.to_f32()))
        }
        _ => Err(PcuTypedConversionReferenceError::UnsupportedProfile),
    }
}

#[allow(clippy::redundant_pub_crate)] // Shared only with the sibling checked CPU executor.
pub(super) const fn store_slice_value(
    slice: &mut PcuCpuTypedSlice<'_>,
    index: usize,
    value: Value,
    binding: PcuBindingRef,
) -> Result<(), PcuTypedConversionReferenceError> {
    match (slice, value) {
        (PcuCpuTypedSlice::ReadWriteI16(slice), Value::I16(value)) => slice[index] = value,
        (PcuCpuTypedSlice::ReadWriteU16(slice), Value::U16(value)) => slice[index] = value,
        (PcuCpuTypedSlice::ReadWriteI32(slice), Value::I32(value)) => slice[index] = value,
        (PcuCpuTypedSlice::ReadWriteU32(slice), Value::U32(value)) => slice[index] = value,
        (PcuCpuTypedSlice::ReadWriteI64(slice), Value::I64(value)) => slice[index] = value,
        (PcuCpuTypedSlice::ReadWriteU64(slice), Value::U64(value)) => slice[index] = value,
        (PcuCpuTypedSlice::ReadWriteF32(slice), Value::F32(value)) => slice[index] = value,
        (PcuCpuTypedSlice::ReadWriteF64(slice), Value::F64(value)) => slice[index] = value,
        (PcuCpuTypedSlice::ReadWriteF16Bits(slice), Value::F16Bits(value)) => slice[index] = value,
        (PcuCpuTypedSlice::ReadWriteBf16Bits(slice), Value::Bf16Bits(value)) => {
            slice[index] = value;
        }
        _ => return Err(PcuTypedConversionReferenceError::AccessMismatch(binding)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;
    use std::boxed::Box;

    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchConversion,
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchFeatureCaps,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchSubmission,
        PcuDispatchValueId,
        PcuInvocationShape,
        PcuKernelId,
        PcuScalarType,
        PcuF16Bits,
        PcuBf16Bits,
        PcuValueType,
        PcuValueTypeCaps,
    };

    #[rustfmt::skip]
    use super::{
        PcuCpuTypedBinding,
        PcuCpuTypedSlice,
        PcuTypedConversionReference,
        PcuTypedConversionReferenceError as Error,
    };

    fn scalar_conversion_kernel(
        conversion: PcuDispatchConversion,
        source: PcuScalarType,
        target: PcuScalarType,
        count: u32,
    ) -> PcuDispatchKernelIr<'static> {
        let bindings = Box::leak(Box::new([
            binding(0, source, PcuBindingAccess::ReadOnly),
            binding(1, target, PcuBindingAccess::WriteOnly),
        ]));
        let ops = Box::leak(Box::new([
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(0),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                result: PcuDispatchValueId(1),
                value: PcuDispatchValueId(0),
                conversion,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
        ]));
        PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(10),
            entry: PcuDispatchEntryPoint {
                name: "scalar_conversion",
                logical_shape: [count, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps: PcuValueTypeCaps::for_scalar(source)
                .union(PcuValueTypeCaps::for_scalar(target))
                .union(PcuValueTypeCaps::SCALAR_VALUES),
            feature_caps: PcuDispatchFeatureCaps::default(),
        }
    }

    #[test]
    fn f32_to_f64_exact_cpu_oracle_preserves_signaling_nan_bits() {
        let kernel = scalar_conversion_kernel(
            PcuDispatchConversion::F32ToF64Exact,
            PcuScalarType::F32,
            PcuScalarType::F64,
            7,
        );
        let input = [
            0x0000_0000_u32,
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x7f80_0000,
            0x7f80_0001,
            0xffc1_2345,
        ]
        .map(f32::from_bits);
        let mut output = [0.0_f64; 7];
        let mut bindings = [
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuCpuTypedSlice::ReadF32(&input),
            },
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuCpuTypedSlice::ReadWriteF64(&mut output),
            },
        ];
        PcuTypedConversionReference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(7).unwrap()),
                },
                &mut bindings,
            )
            .unwrap();
        assert_eq!(
            output.map(f64::to_bits),
            input.map(|value| fusion_pcu::f32_bits_to_f64_bits(value.to_bits()))
        );
    }

    fn binding(index: u32, scalar: PcuScalarType, access: PcuBindingAccess) -> PcuBinding<'static> {
        PcuBinding::value(
            None,
            0,
            index,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    }

    fn conversion_kernel(grid: bool, signed: bool, extent: u32) -> PcuDispatchKernelIr<'static> {
        let (source, target, conversion) = if signed {
            (
                PcuScalarType::I8,
                PcuScalarType::I16,
                PcuDispatchConversion::I8ToI16,
            )
        } else {
            (
                PcuScalarType::U8,
                PcuScalarType::U16,
                PcuDispatchConversion::U8ToU16,
            )
        };
        integer_conversion_kernel(grid, source, target, conversion, extent)
    }

    fn integer_conversion_kernel(
        grid: bool,
        source: PcuScalarType,
        target: PcuScalarType,
        conversion: PcuDispatchConversion,
        extent: u32,
    ) -> PcuDispatchKernelIr<'static> {
        let bindings = Box::leak(Box::new([
            binding(0, source, PcuBindingAccess::ReadOnly),
            binding(1, target, PcuBindingAccess::WriteOnly),
        ]));
        let index = if grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = Box::leak(Box::new([
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(0),
                binding: PcuBindingRef::new(0, 0),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Convert {
                result: PcuDispatchValueId(1),
                value: PcuDispatchValueId(0),
                conversion,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index,
                value: PcuDispatchValueId(1),
            }),
        ]));
        let ops = if grid {
            Box::leak(Box::new([
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
            ])) as &'static [PcuDispatchOp<'static>]
        } else {
            Box::leak(Box::new([
                body[0],
                body[1],
                body[2],
                PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
            ]))
        };
        let type_caps = PcuValueTypeCaps::for_scalar(source)
            .union(PcuValueTypeCaps::for_scalar(target))
            .union(PcuValueTypeCaps::SCALAR_VALUES);
        PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(9),
            entry: PcuDispatchEntryPoint {
                name: "typed_conversion",
                logical_shape: [if grid { 3 } else { extent }, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps,
            feature_caps: PcuDispatchFeatureCaps::default(),
        }
    }

    #[test]
    fn signed_and_unsigned_conversions_preserve_exact_values_direct() {
        let signed_kernel = conversion_kernel(false, true, 5);
        let signed = [-128_i8, -1, 0, 1, 127];
        let mut signed_out = [0_i16; 5];
        let mut signed_bindings = [
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuCpuTypedSlice::ReadI8(&signed),
            },
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuCpuTypedSlice::ReadWriteI16(&mut signed_out),
            },
        ];
        PcuTypedConversionReference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &signed_kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(5).unwrap()),
                },
                &mut signed_bindings,
            )
            .unwrap();
        assert_eq!(signed_out, [-128, -1, 0, 1, 127]);

        let unsigned_kernel = conversion_kernel(false, false, 5);
        let unsigned = [0_u8, 1, 127, 128, 255];
        let mut unsigned_out = [0_u16; 5];
        let mut unsigned_bindings = [
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuCpuTypedSlice::ReadU8(&unsigned),
            },
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuCpuTypedSlice::ReadWriteU16(&mut unsigned_out),
            },
        ];
        PcuTypedConversionReference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &unsigned_kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(5).unwrap()),
                },
                &mut unsigned_bindings,
            )
            .unwrap();
        assert_eq!(unsigned_out, [0, 1, 127, 128, 255]);
    }

    macro_rules! check_widening {
        ($source:expr, $target:ty, $source_scalar:expr, $target_scalar:expr, $conversion:expr, $read:ident, $write:ident, $expected:expr) => {{
            let input = $source;
            let expected: [$target; 7] = $expected;
            let mut direct_output = [0 as $target; 7];
            let direct_kernel =
                integer_conversion_kernel(false, $source_scalar, $target_scalar, $conversion, 7);
            let mut direct_bindings = [
                PcuCpuTypedBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuCpuTypedSlice::$read(&input),
                },
                PcuCpuTypedBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuCpuTypedSlice::$write(&mut direct_output),
                },
            ];
            PcuTypedConversionReference
                .run_host_direct(
                    PcuDispatchSubmission {
                        kernel: &direct_kernel,
                        shape: PcuInvocationShape::invocations(NonZeroU32::new(7).unwrap()),
                    },
                    &mut direct_bindings,
                )
                .unwrap();
            assert_eq!(direct_output, expected);

            let mut grid_output = [0 as $target; 7];
            let grid_kernel =
                integer_conversion_kernel(true, $source_scalar, $target_scalar, $conversion, 7);
            let mut grid_bindings = [
                PcuCpuTypedBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuCpuTypedSlice::$read(&input),
                },
                PcuCpuTypedBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuCpuTypedSlice::$write(&mut grid_output),
                },
            ];
            PcuTypedConversionReference
                .run_host_direct(
                    PcuDispatchSubmission {
                        kernel: &grid_kernel,
                        shape: PcuInvocationShape::invocations(NonZeroU32::new(3).unwrap()),
                    },
                    &mut grid_bindings,
                )
                .unwrap();
            assert_eq!(grid_output, expected);
        }};
    }

    #[test]
    fn wider_integer_conversions_preserve_boundaries_direct_and_grid_stride() {
        check_widening!(
            [i16::MIN, -1, 0, 1, i16::MAX, -12345, 23456],
            i32,
            PcuScalarType::I16,
            PcuScalarType::I32,
            PcuDispatchConversion::I16ToI32,
            ReadI16,
            ReadWriteI32,
            [
                i32::from(i16::MIN),
                -1,
                0,
                1,
                i32::from(i16::MAX),
                -12345,
                23456
            ]
        );
        check_widening!(
            [0_u16, 1, 0x7fff, 0x8000, u16::MAX, 12345, 54321],
            u32,
            PcuScalarType::U16,
            PcuScalarType::U32,
            PcuDispatchConversion::U16ToU32,
            ReadU16,
            ReadWriteU32,
            [0, 1, 0x7fff, 0x8000, u32::from(u16::MAX), 12345, 54321]
        );
        check_widening!(
            [i32::MIN, -1, 0, 1, i32::MAX, -1_234_567_890, 1_234_567_890],
            i64,
            PcuScalarType::I32,
            PcuScalarType::I64,
            PcuDispatchConversion::I32ToI64,
            ReadI32,
            ReadWriteI64,
            [
                i64::from(i32::MIN),
                -1,
                0,
                1,
                i64::from(i32::MAX),
                -1_234_567_890,
                1_234_567_890
            ]
        );
        check_widening!(
            [
                0_u32,
                1,
                0x7fff_ffff,
                0x8000_0000,
                u32::MAX,
                1_234_567_890,
                3_000_000_000
            ],
            u64,
            PcuScalarType::U32,
            PcuScalarType::U64,
            PcuDispatchConversion::U32ToU64,
            ReadU32,
            ReadWriteU64,
            [
                0,
                1,
                0x7fff_ffff,
                0x8000_0000,
                u64::from(u32::MAX),
                1_234_567_890,
                3_000_000_000
            ]
        );
    }

    #[test]
    fn grid_stride_conversion_handles_padded_launch_and_tail() {
        let kernel = conversion_kernel(true, false, 8);
        let input = [0_u8, 127, 128, 255, 4, 5, 6, 7];
        let mut output = [u16::MAX; 8];
        let mut bindings = [
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuCpuTypedSlice::ReadU8(&input),
            },
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuCpuTypedSlice::ReadWriteU16(&mut output),
            },
        ];
        PcuTypedConversionReference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(3).unwrap()),
                },
                &mut bindings,
            )
            .unwrap();
        assert_eq!(output, [0, 127, 128, 255, 4, 5, 6, 7]);
    }

    #[test]
    fn rejects_wrong_host_type_before_writing_output() {
        let kernel = conversion_kernel(false, true, 2);
        let source = [1_u8, 2];
        let mut output = [99_i16; 2];
        let mut bindings = [
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuCpuTypedSlice::ReadU8(&source),
            },
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuCpuTypedSlice::ReadWriteI16(&mut output),
            },
        ];
        let result = PcuTypedConversionReference.run_host_direct(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(2).unwrap()),
            },
            &mut bindings,
        );
        assert_eq!(result, Err(Error::TypeMismatch(PcuBindingRef::new(0, 0))));
        assert_eq!(output, [99, 99]);
    }

    #[test]
    fn f32_to_f16_cpu_oracle_obeys_rounding_overflow_subnormal_and_nan_rules() {
        let inputs = [
            0.0_f32,
            -0.0,
            1.0,
            65_504.0,
            65_520.0,
            2.0_f32.powi(-24),
            2.0_f32.powi(-25),
            f32::from_bits(0xffc1_2345),
        ];
        let mut output = [PcuF16Bits::from_bits(0); 8];
        let kernel = scalar_conversion_kernel(
            PcuDispatchConversion::F32ToF16Bits,
            PcuScalarType::F32,
            PcuScalarType::F16,
            8,
        );
        let mut bindings = [
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuCpuTypedSlice::ReadF32(&inputs),
            },
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuCpuTypedSlice::ReadWriteF16Bits(&mut output),
            },
        ];
        PcuTypedConversionReference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(8).unwrap()),
                },
                &mut bindings,
            )
            .unwrap();
        assert_eq!(
            output.map(PcuF16Bits::to_bits),
            [0, 0x8000, 0x3c00, 0x7bff, 0x7c00, 1, 0, 0xfe09]
        );
    }

    #[test]
    fn f16_to_f32_cpu_oracle_preserves_exact_widening_including_signaling_nan() {
        let inputs = [
            PcuF16Bits::from_bits(0),
            PcuF16Bits::from_bits(0x8000),
            PcuF16Bits::from_bits(1),
            PcuF16Bits::from_bits(0x3c00),
            PcuF16Bits::from_bits(0x7c01),
        ];
        let mut output = [0.0_f32; 5];
        let kernel = scalar_conversion_kernel(
            PcuDispatchConversion::F16BitsToF32,
            PcuScalarType::F16,
            PcuScalarType::F32,
            5,
        );
        let mut bindings = [
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuCpuTypedSlice::ReadF16Bits(&inputs),
            },
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuCpuTypedSlice::ReadWriteF32(&mut output),
            },
        ];
        PcuTypedConversionReference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(5).unwrap()),
                },
                &mut bindings,
            )
            .unwrap();
        assert_eq!(
            output.map(f32::to_bits),
            [0, 0x8000_0000, 0x3380_0000, 0x3f80_0000, 0x7f80_2000]
        );
    }

    #[test]
    fn bf16_narrow_and_widen_use_ties_even_and_preserve_signaling_nan_on_widen() {
        let inputs = [
            1.0_f32,
            f32::from_bits(1.0_f32.to_bits() + 0x0000_8000),
            f32::from_bits(1.0_f32.to_bits() + 0x0001_8000),
            -0.0,
            f32::from_bits(0xff81_2345),
        ];
        let mut narrowed = [PcuBf16Bits::from_bits(0); 5];
        let narrow_kernel = scalar_conversion_kernel(
            PcuDispatchConversion::F32ToBf16Bits,
            PcuScalarType::F32,
            PcuScalarType::BF16,
            5,
        );
        let mut narrow_bindings = [
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuCpuTypedSlice::ReadF32(&inputs),
            },
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuCpuTypedSlice::ReadWriteBf16Bits(&mut narrowed),
            },
        ];
        PcuTypedConversionReference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &narrow_kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(5).unwrap()),
                },
                &mut narrow_bindings,
            )
            .unwrap();
        assert_eq!(
            narrowed.map(PcuBf16Bits::to_bits),
            [0x3f80, 0x3f80, 0x3f82, 0x8000, 0xffc1]
        );

        let widen_kernel = scalar_conversion_kernel(
            PcuDispatchConversion::Bf16BitsToF32,
            PcuScalarType::BF16,
            PcuScalarType::F32,
            2,
        );
        let narrow_inputs = [
            PcuBf16Bits::from_bits(0x0001),
            PcuBf16Bits::from_bits(0x7f81),
        ];
        let mut widened = [0.0_f32; 2];
        let mut widen_bindings = [
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuCpuTypedSlice::ReadBf16Bits(&narrow_inputs),
            },
            PcuCpuTypedBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuCpuTypedSlice::ReadWriteF32(&mut widened),
            },
        ];
        PcuTypedConversionReference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &widen_kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(2).unwrap()),
                },
                &mut widen_bindings,
            )
            .unwrap();
        assert_eq!(widened.map(f32::to_bits), [0x0001_0000, 0x7f81_0000]);
    }
}

#[rustfmt::skip]
use fusion_pcu::{
    validate_f16_identity_kernel,
    validate_bf16_identity_kernel,
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
    PcuF16Bits,
    PcuBf16Bits,
};

#[rustfmt::skip]
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

/// Explicit unchecked host-arithmetic oracle for the bounded `f32` indexed-map profile.
/// Use [`crate::PcuCheckedF32Reference`] for checked arithmetic and terminal fault semantics.
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

/// Explicit unchecked host-arithmetic oracle for direct and grid-stride f64 maps, including min/max.
/// Use [`crate::PcuCheckedF64Reference`] for checked arithmetic and terminal fault semantics.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuF64Reference;

/// CPU transport oracle for binary16 payloads. Values are copied as opaque bits.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuF16IdentityReference;
/// CPU transport oracle for bfloat16 payloads. Values are copied as opaque bits.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuBf16IdentityReference;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuHalfIdentityReferenceError {
    InvalidSubmission,
    InvalidKernel,
}

macro_rules! impl_half_identity {
    ($ty:ty, $validator:ident, $name:ty) => {
        // SAFETY: Copy runs synchronously and never retains a host reference.
        unsafe impl PcuSynchronousHostDispatchBackend<$ty> for $name {
            type Error = PcuHalfIdentityReferenceError;
            fn run_host_direct(
                &self,
                submission: PcuDispatchSubmission<'_>,
                bindings: &mut [PcuHostScalarBinding<'_, $ty>],
                _parameters: PcuInvocationParameters<'_>,
            ) -> Result<(), Self::Error> {
                validate_host_scalar_bindings::<$ty, ()>(submission, bindings)
                    .map_err(|_| PcuHalfIdentityReferenceError::InvalidSubmission)?;
                $validator(submission.kernel)
                    .map_err(|_| PcuHalfIdentityReferenceError::InvalidKernel)?;
                let width = submission.shape.invocation_count().get() as usize;
                let (extent, grid_body) = match submission.kernel.ops.first() {
                    Some(PcuDispatchOp::GridStrideLoop { extent, body }) => {
                        (*extent as usize, Some(*body))
                    }
                    _ => (width, None),
                };
                for invocation in 0..width {
                    let mut index = invocation;
                    while index < extent {
                        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            binding: input,
                            ..
                        }) = grid_body.map_or(submission.kernel.ops[0], |body| body[0])
                        else {
                            unreachable!()
                        };
                        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding: output,
                            ..
                        }) = grid_body.map_or(submission.kernel.ops[1], |body| body[1])
                        else {
                            unreachable!()
                        };
                        let src = bindings
                            .iter()
                            .find(|b| b.target == input)
                            .expect("validated input binding");
                        let value = match &src.slice {
                            PcuHostScalarSlice::Read(s) => s[index],
                            PcuHostScalarSlice::ReadWrite(s) => s[index],
                        };
                        let dst = bindings
                            .iter_mut()
                            .find(|b| b.target == output)
                            .expect("validated output binding");
                        let PcuHostScalarSlice::ReadWrite(s) = &mut dst.slice else {
                            unreachable!("validated write binding")
                        };
                        s[index] = value;
                        if extent - index <= width {
                            break;
                        }
                        index += width;
                    }
                }
                Ok(())
            }
        }
    };
}

impl_half_identity!(
    PcuF16Bits,
    validate_f16_identity_kernel,
    PcuF16IdentityReference
);
impl_half_identity!(
    PcuBf16Bits,
    validate_bf16_identity_kernel,
    PcuBf16IdentityReference
);

#[cfg(test)]
mod half_identity_tests {
    use core::num::NonZeroU32;
    use std::boxed::Box;
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchValueId,
        PcuF16Bits,
        PcuBf16Bits,
        PcuInvocationParameters,
        PcuInvocationShape,
        PcuHostScalarBinding,
        PcuHostScalarSlice,
        PcuDispatchSubmission,
        PcuValueType,
        PcuValueTypeCaps,
    };
    #[rustfmt::skip]
    use super::{
        PcuBf16IdentityReference,
        PcuF16IdentityReference,
    };
    use fusion_pcu::PcuSynchronousHostDispatchBackend;

    fn ir(
        scalar: fusion_pcu::PcuScalarType,
        grid: bool,
    ) -> ([PcuBinding<'static>; 2], PcuDispatchKernelIr<'static>) {
        let value_type = PcuValueType::Scalar(scalar);
        let bindings = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                value_type,
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                value_type,
            ),
        ];
        let ops: &'static [PcuDispatchOp<'static>] = if grid {
            let body = Box::leak(Box::new([
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::GridStrideId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::GridStrideId,
                    value: PcuDispatchValueId(1),
                }),
            ]));
            Box::leak(Box::new([
                PcuDispatchOp::GridStrideLoop { extent: 5, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ]))
        } else {
            Box::leak(Box::new([
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(1),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ]))
        };
        let kernel_bindings: &'static [PcuBinding<'static>] = Box::leak(Box::new(bindings));
        let kernel = PcuDispatchKernelIr {
            id: fusion_pcu::PcuKernelId(1),
            entry: fusion_pcu::PcuDispatchEntryPoint {
                name: "half_identity",
                logical_shape: [if grid { 2 } else { 5 }, 1, 1],
            },
            bindings: kernel_bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: fusion_pcu::PcuDispatchFeatureCaps::empty(),
        };
        (bindings, kernel)
    }

    #[test]
    fn f16_and_bf16_copy_payloads_exactly_in_direct_and_grid_profiles() {
        let bits = [0x0001, 0x8000, 0x7c01, 0x7e55, 0xffff];
        let source: [PcuF16Bits; 5] = bits.map(PcuF16Bits::from_bits);
        for grid in [false, true] {
            let (bindings, kernel) = ir(fusion_pcu::PcuScalarType::F16, grid);
            let mut out = [PcuF16Bits::from_bits(0); 5];
            let mut host = [
                PcuHostScalarBinding {
                    target: bindings[0].reference(),
                    slice: PcuHostScalarSlice::Read(&source),
                },
                PcuHostScalarBinding {
                    target: bindings[1].reference(),
                    slice: PcuHostScalarSlice::ReadWrite(&mut out),
                },
            ];
            let shape =
                PcuInvocationShape::invocations(NonZeroU32::new(if grid { 2 } else { 5 }).unwrap());
            PcuF16IdentityReference
                .run_host_direct(
                    PcuDispatchSubmission {
                        kernel: &kernel,
                        shape,
                    },
                    &mut host,
                    PcuInvocationParameters::empty(),
                )
                .unwrap();
            assert_eq!(out.map(PcuF16Bits::to_bits), bits);
        }
        let (bindings, kernel) = ir(fusion_pcu::PcuScalarType::BF16, true);
        let source = bits.map(PcuBf16Bits::from_bits);
        let mut out = [PcuBf16Bits::from_bits(0); 5];
        let mut host = [
            PcuHostScalarBinding {
                target: bindings[0].reference(),
                slice: PcuHostScalarSlice::Read(&source),
            },
            PcuHostScalarBinding {
                target: bindings[1].reference(),
                slice: PcuHostScalarSlice::ReadWrite(&mut out),
            },
        ];
        PcuBf16IdentityReference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(2).unwrap()),
                },
                &mut host,
                PcuInvocationParameters::empty(),
            )
            .unwrap();
        assert_eq!(out.map(PcuBf16Bits::to_bits), bits);
    }
}

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
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                let PcuParameterValue::F64(bits) = value else {
                    unreachable!("f64 validator checks constant width");
                };
                values[usize::from(result.0)] = Some(f64::from_bits(*bits));
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
                    PcuDispatchAluOp::Min => f64_min(left, right),
                    PcuDispatchAluOp::Max => f64_max(left, right),
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

const fn f64_min(left: f64, right: f64) -> f64 {
    if left == 0.0 && right == 0.0 {
        f64::from_bits(1_u64 << 63)
    } else {
        left.min(right)
    }
}

const fn f64_max(left: f64, right: f64) -> f64 {
    if left == 0.0 && right == 0.0 {
        0.0
    } else {
        left.max(right)
    }
}

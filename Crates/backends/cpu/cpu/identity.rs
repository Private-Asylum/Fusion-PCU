//! Representation-preserving identity transport for every sealed scalar type.

#[rustfmt::skip]
use fusion_pcu::{
    validate_host_scalar_bindings,
    validate_parameters,
    validate_scalar_identity_kernel,
    validate_scalar_broadcast_kernel,
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuKernelIrContract,
    PcuScalar,
    PcuScalarIdentityValidationError,
    PcuSynchronousHostDispatchBackend,
};

/// Failure to execute a typed scalar identity copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuScalarIdentityReferenceError {
    UnsupportedNumericalRequirements,
    InvalidSubmission,
    InvalidKernel(PcuScalarIdentityValidationError),
    MissingBinding(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
}

/// Opt-in synchronous CPU oracle for scalar transport, without arithmetic or numeric conversion.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuScalarIdentityReference;

// SAFETY: All shapes, accesses and IR are admitted before indexing caller-owned slices.
// Execution is synchronous and retains no host references after return, including on error.
unsafe impl<T: PcuScalar> PcuSynchronousHostDispatchBackend<T> for PcuScalarIdentityReference {
    type Error = PcuScalarIdentityReferenceError;

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
            return Err(PcuScalarIdentityReferenceError::UnsupportedNumericalRequirements);
        }
        validate_host_scalar_bindings::<T, ()>(submission, bindings)
            .map_err(|_| PcuScalarIdentityReferenceError::InvalidSubmission)?;
        validate_parameters(submission.kernel.signature(), parameters)
            .map_err(|_| PcuScalarIdentityReferenceError::InvalidSubmission)?;

        let width = submission.shape.invocation_count().get() as usize;
        let (body, extent) = match submission.kernel.ops {
            [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (*body, *extent as usize),
            ops => (ops, width),
        };
        let broadcast = matches!(
            body.first(),
            Some(PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                index: PcuDispatchIndex::BindingElementZero,
                ..
            }))
        );
        let validation = if broadcast {
            validate_scalar_broadcast_kernel(submission.kernel, T::TYPE)
        } else {
            validate_scalar_identity_kernel(submission.kernel, T::TYPE)
        };
        validation.map_err(PcuScalarIdentityReferenceError::InvalidKernel)?;
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding: input, .. }) = body[0]
        else {
            unreachable!("validated identity begins with one load")
        };
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output, ..
        }) = body[1]
        else {
            unreachable!("validated identity ends with one store")
        };
        let input_index = bindings
            .iter()
            .position(|binding| binding.target == input)
            .ok_or(PcuScalarIdentityReferenceError::MissingBinding(input))?;
        let output_index = bindings
            .iter()
            .position(|binding| binding.target == output)
            .ok_or(PcuScalarIdentityReferenceError::MissingBinding(output))?;
        // The admitted profile only repeats or copies representation bits. A nonzero grid stride
        // covers exactly the same prefix once; it needs no per-invocation interpreter loop.
        if input_index < output_index {
            let (before, after) = bindings.split_at_mut(output_index);
            copy_prefix(&before[input_index], &mut after[0], extent, broadcast)?;
        } else {
            let (before, after) = bindings.split_at_mut(input_index);
            copy_prefix(&after[0], &mut before[output_index], extent, broadcast)?;
        }
        Ok(())
    }
}

fn copy_prefix<T: PcuScalar>(
    source: &PcuHostScalarBinding<'_, T>,
    destination: &mut PcuHostScalarBinding<'_, T>,
    extent: usize,
    broadcast: bool,
) -> Result<(), PcuScalarIdentityReferenceError> {
    let source = match &source.slice {
        PcuHostScalarSlice::Read(slice) => *slice,
        PcuHostScalarSlice::ReadWrite(slice) => &**slice,
    };
    let PcuHostScalarSlice::ReadWrite(output) = &mut destination.slice else {
        return Err(PcuScalarIdentityReferenceError::AccessMismatch(
            destination.target,
        ));
    };
    if broadcast {
        output[..extent].fill(source[0]);
    } else {
        output[..extent].copy_from_slice(&source[..extent]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBindingRef,
        PcuDispatchDataOp,
        PcuDispatchOp,
        PcuDispatchSubmission,
        PcuHostScalarBinding,
        PcuHostScalarSlice,
        PcuInvocationParameters,
        PcuScalar,
        PcuSynchronousHostDispatchBackend,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingStorageClass,
        PcuDispatchControlOp,
        PcuDispatchEntryPoint,
        PcuDispatchFeatureCaps,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchValueId,
        PcuF16Bits,
        PcuBf16Bits,
        PcuInvocationShape,
        PcuKernelId,
        PcuValueTypeCaps,
    };
    use super::PcuScalarIdentityReference;

    fn check_transport<T: PcuScalar>(values: [T; 5]) {
        for grid in [false, true] {
            let extent = if grid { 5 } else { 3 };
            let width = if grid { 2 } else { 3 };
            let index = if grid {
                PcuDispatchIndex::GridStrideId
            } else {
                PcuDispatchIndex::InvocationId
            };
            let bindings = [
                PcuBinding::scalar::<T>(
                    Some("input"),
                    0,
                    0,
                    PcuBindingStorageClass::Storage,
                    PcuBindingAccess::ReadOnly,
                ),
                PcuBinding::scalar::<T>(
                    Some("output"),
                    0,
                    1,
                    PcuBindingStorageClass::Storage,
                    PcuBindingAccess::WriteOnly,
                ),
            ];
            let body = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index,
                    value: PcuDispatchValueId(1),
                }),
            ];
            let ops = [
                body[0],
                body[1],
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let grid_ops = [
                PcuDispatchOp::GridStrideLoop {
                    extent: 5,
                    body: &body,
                },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = PcuDispatchKernelIr {
                numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                id: PcuKernelId(1),
                entry: PcuDispatchEntryPoint {
                    name: "identity",
                    logical_shape: [width, 1, 1],
                },
                bindings: &bindings,
                ports: &[],
                parameters: &[],
                ops: if grid { &grid_ops } else { &ops },
                type_caps: PcuValueTypeCaps::for_scalar(T::TYPE),
                feature_caps: PcuDispatchFeatureCaps::default(),
            };
            let mut output = [values[0]; 6];
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&values),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::ReadWrite(&mut output),
                },
            ];
            if grid {
                host.swap(0, 1);
            }
            PcuScalarIdentityReference
                .run_host_direct(
                    PcuDispatchSubmission {
                        kernel: &kernel,
                        shape: PcuInvocationShape::invocations(NonZeroU32::new(width).unwrap()),
                    },
                    &mut host,
                    PcuInvocationParameters::empty(),
                )
                .unwrap();
            for (index, actual) in output.iter().enumerate() {
                let expected = if index < extent {
                    values[index]
                } else {
                    values[0]
                };
                assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
            }
        }
    }

    #[test]
    fn identity_preserves_all_scalar_representations_and_unwritten_tails() {
        check_transport([0_u8, u8::MAX, 1, 128, 7]);
        check_transport([0_u16, u16::MAX, 1, 0x8000, 7]);
        check_transport([0_u32, u32::MAX, 1, 0x8000_0000, 7]);
        check_transport([0_u64, u64::MAX, 1, 0x8000_0000_0000_0000, 7]);
        check_transport([0_i8, i8::MIN, i8::MAX, -1, 7]);
        check_transport([0_i16, i16::MIN, i16::MAX, -1, 7]);
        check_transport([0_i32, i32::MIN, i32::MAX, -1, 7]);
        check_transport([0_i64, i64::MIN, i64::MAX, -1, 7]);
        check_transport([
            0.0_f32,
            -0.0,
            f32::from_bits(1),
            f32::INFINITY,
            f32::from_bits(0x7fc1_2345),
        ]);
        check_transport([
            0.0_f64,
            -0.0,
            f64::from_bits(1),
            f64::INFINITY,
            f64::from_bits(0x7ff8_1234_5678_9abc),
        ]);
        check_transport([0_u16, 0x8000, 1, 0x7c00, 0x7e12].map(PcuF16Bits::from_bits));
        check_transport([0_u16, 0x8000, 1, 0x7f80, 0x7fc1].map(PcuBf16Bits::from_bits));
    }
}

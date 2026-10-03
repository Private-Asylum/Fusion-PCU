use super::*;
fn policy(
    operation: PcuSpirvCompoundOperation,
    underflow: PcuFloatUnderflowPolicy,
) -> CompoundStatusPolicy {
    let TensorStatusPolicy::Compound(policy) =
        TensorStatusPolicy::compound(PcuScalarType::F32, operation, underflow).unwrap()
    else {
        panic!("compound factory");
    };
    policy
}
fn matmul(inner: u32, underflow: PcuFloatUnderflowPolicy) -> CompoundStatusPolicy {
    policy(
        PcuSpirvCompoundOperation::MatMul {
            rows: 2,
            inner,
            columns: 1,
            transpose_left: false,
            transpose_right: false,
        },
        underflow,
    )
}
fn scan(
    policy: CompoundStatusPolicy,
    records: &[[u32; 3]],
) -> Result<Option<VulkanCompoundFault>, PcuVulkanError> {
    policy.scan(records.len(), |index| Ok(records[index]))
}
#[test]
fn matmul_validates_reduction_step_kind_and_every_later_record() {
    let p = matmul(3, PcuFloatUnderflowPolicy::IeeeAfterRounding);
    let result = scan(p, &[[3, 2, 1], [3, 0, 0]]).unwrap().unwrap();
    assert_eq!(result.element_index, 0);
    assert_eq!(result.reduction_index, 2);
    assert_eq!(result.step, TensorArithmeticStep::Add);
    for invalid in [
        [3, 3, 0],
        [3, 0, 2],
        [4, 0, 0],
        [2, 0, 1],
        [6, 0, 0],
        [3, 0, 4],
    ] {
        assert!(matches!(
            scan(p, &[[3, 0, 0], invalid]),
            Err(PcuVulkanError::InvalidCompoundStatus {
                element_index: 1,
                ..
            })
        ));
    }
}
#[test]
fn sgd_uses_only_zero_reduction_and_multiply_subtract_steps() {
    let p = policy(
        PcuSpirvCompoundOperation::Sgd {
            count: 2,
            learning_rate: 0.5,
        },
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    );
    assert!(matches!(
        scan(p, &[[0, 0, 0], [2, 0, 2]]),
        Ok(Some(VulkanCompoundFault {
            element_index: 1,
            step: TensorArithmeticStep::Subtract,
            ..
        }))
    ));
    for invalid in [[3, 1, 0], [3, 0, 1], [2, 0, 3], [4, 0, 2]] {
        assert!(matches!(
            scan(p, &[invalid]),
            Err(PcuVulkanError::InvalidCompoundStatus { .. })
        ));
    }
}
#[test]
fn mean_division_uses_final_count_coordinate_and_only_policy_underflow() {
    for underflow in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        let p = policy(
            PcuSpirvCompoundOperation::MeanSquaredError { count: 3 },
            underflow,
        );
        let result = scan(p, &[[2, 3, 3]]);
        if underflow == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
            assert!(matches!(
                result,
                Err(PcuVulkanError::InvalidCompoundStatus { .. })
            ));
        } else {
            assert!(matches!(
                result,
                Ok(Some(VulkanCompoundFault {
                    reduction_index: 3,
                    step: TensorArithmeticStep::Divide,
                    ..
                }))
            ));
        }
        for invalid in [
            [1, 3, 3],
            [3, 3, 3],
            [4, 3, 3],
            [2, 4, 3],
            [3, 3, 0],
            [3, 3, 2],
        ] {
            assert!(matches!(
                scan(p, &[invalid]),
                Err(PcuVulkanError::InvalidCompoundStatus { .. })
            ));
        }
    }
}
#[test]
fn empty_matmul_has_zero_arithmetic_domain_without_rejecting_zero_outputs() {
    let p = matmul(0, PcuFloatUnderflowPolicy::IeeeAfterRounding);
    assert_eq!(p.domain.event_extent(), 0);
    assert!(
        scan(p, &[[0, u32::MAX, u32::MAX], [0, 0, 0]])
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        scan(p, &[[3, 0, 0], [0, 0, 0]]),
        Err(PcuVulkanError::InvalidCompoundStatus { .. })
    ));
    assert!(matches!(
        TensorStatusPolicy::compound(
            PcuScalarType::F32,
            PcuSpirvCompoundOperation::MeanSquaredError { count: 0 },
            PcuFloatUnderflowPolicy::IeeeAfterRounding
        ),
        Err(PcuVulkanError::UnsupportedPreparedProfile)
    ));
}
#[test]
fn valid_fatal_does_not_stop_physical_record_validation() {
    let p = matmul(1, PcuFloatUnderflowPolicy::IeeeAfterRounding);
    let mut visited = 0;
    let fault = p
        .scan(2, |index| {
            visited += 1;
            Ok(if index == 0 { [3, 0, 0] } else { [0, 0, 0] })
        })
        .unwrap()
        .unwrap();
    assert_eq!(fault.element_index, 0);
    assert_eq!(visited, 2);
    assert!(matches!(
        p.scan(2, |index| if index == 0 {
            Ok([3, 0, 0])
        } else {
            Err(PcuVulkanError::Quarantined)
        }),
        Err(PcuVulkanError::Quarantined)
    ));
}

#[test]
fn dependent_steps_reject_classes_excluded_by_successful_prior_steps() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let TensorStatusPolicy::Compound(matmul) = TensorStatusPolicy::compound(
                scalar,
                PcuSpirvCompoundOperation::MatMul {
                    rows: 1,
                    inner: 2,
                    columns: 1,
                    transpose_left: false,
                    transpose_right: false,
                },
                underflow,
            )
            .unwrap() else {
                panic!("compound factory");
            };
            // The first Add is +0 plus the finite checked product. Later Add
            // operands are finite because earlier Multiply/Add already succeeded.
            for invalid in [[1, 0, 1], [2, 0, 1], [3, 0, 1], [1, 1, 1]] {
                assert!(matches!(
                    scan(matmul, &[invalid]),
                    Err(PcuVulkanError::InvalidCompoundStatus { .. })
                ));
            }
            assert!(scan(matmul, &[[3, 1, 1]]).unwrap().is_some());
            let TensorStatusPolicy::Compound(mse) = TensorStatusPolicy::compound(
                scalar,
                PcuSpirvCompoundOperation::MeanSquaredError { count: 2 },
                underflow,
            )
            .unwrap() else {
                panic!("compound factory");
            };
            // Checked Subtract supplies a finite Square input. Accumulation
            // adds nonnegative checked squares and cannot produce underflow.
            for invalid in [[1, 0, 0], [1, 1, 1], [2, 1, 1]] {
                assert!(matches!(
                    scan(mse, &[invalid]),
                    Err(PcuVulkanError::InvalidCompoundStatus { .. })
                ));
            }
        }
    }
}

#[test]
#[ignore = "requires an actual Vulkan device; deterministic structured-status protocol exercise, not a driver failure"]
fn native_structured_records_enforce_actual_reduction_domain() {
    #[rustfmt::skip]
    use crate::ffi::{
        VulkanDevice,
        VulkanOwnedBuffer,
    };
    let device = std::rc::Rc::new(VulkanDevice::new().unwrap());
    let mut status = VulkanOwnedBuffer::new(&device, 24).unwrap();
    let policy = matmul(3, PcuFloatUnderflowPolicy::IeeeAfterRounding);
    for records in [
        [[3_u32, 0, 0], [3, 3, 0]],
        [[3, 0, 0], [4, 0, 0]],
        [[3, 0, 0], [3, 0, 3]],
    ] {
        let mut bytes = [0; 24];
        for (slot, code) in bytes
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(records.into_iter().flatten())
        {
            slot.copy_from_slice(&code.to_ne_bytes());
        }
        // These host-authored physical records exercise the production structured decoder
        // using actual coherent native storage; no shader or driver failure is manufactured.
        status.write(&bytes).unwrap();
        assert!(matches!(
            policy.scan(2, |element| Ok([
                status.status(element * 3)?,
                status.status(element * 3 + 1)?,
                status.status(element * 3 + 2)?
            ])),
            Err(PcuVulkanError::InvalidCompoundStatus {
                element_index: 1,
                ..
            })
        ));
    }
}

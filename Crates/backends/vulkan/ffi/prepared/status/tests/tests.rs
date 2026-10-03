use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuFloatUnderflowPolicy,
    model::{PcuDispatchFloatBinaryOp, PcuDispatchIntegerBinaryOp},
};

fn scan(
    policy: StatusPolicy,
    records: &[u32],
) -> Result<Option<PcuExecutionFault>, PcuVulkanError> {
    policy.scan(records.len(), |index| Ok(records[index]))
}
fn integer(
    scalar: PcuScalarType,
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
) -> StatusPolicy {
    StatusPolicy::checked(
        PcuCheckedScalarFaultLaw::integer_binary(scalar, op, range),
        range,
    )
    .unwrap()
}
#[test]
fn transport_accepts_only_zero_records() {
    assert_eq!(scan(StatusPolicy::ZERO_ONLY, &[0, 0]).unwrap(), None);
    for code in [1, 2, 3, 4, 5, 6, 255, u32::MAX] {
        assert!(
            matches!(scan(StatusPolicy::ZERO_ONLY, &[0, code]), Err(PcuVulkanError::InvalidStatus { code: actual, invocation_id: 1 }) if actual == code)
        );
    }
}
#[test]
fn every_record_is_validated_before_fault_arbitration() {
    let policy = integer(
        PcuScalarType::U32,
        PcuDispatchIntegerBinaryOp::Add,
        PcuRangePolicy::Reject,
    );
    let mut visited = 0;
    assert!(matches!(
        policy.scan(4, |index| {
            visited += 1;
            Ok([0, 3, 0, 0][index])
        }),
        Err(PcuVulkanError::Fault(PcuExecutionFault {
            invocation_id: 1,
            ..
        }))
    ));
    assert_eq!(visited, 4);
    for code in [1, 2, 4, 5, 6, u32::MAX] {
        assert!(matches!(
            scan(policy, &[3, 0, code]),
            Err(PcuVulkanError::InvalidStatus {
                invocation_id: 2,
                ..
            })
        ));
    }
}
#[test]
fn recovered_notices_and_fatal_priority_are_exact() {
    let policy = StatusPolicy::checked(
        PcuCheckedScalarFaultLaw::float_binary(
            PcuScalarType::F32,
            PcuDispatchFloatBinaryOp::Div,
            PcuRangePolicy::Clamp,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ),
        PcuRangePolicy::Clamp,
    )
    .unwrap();
    assert!(matches!(
        scan(policy, &[0, 2, 3]),
        Ok(Some(PcuExecutionFault {
            invocation_id: 1,
            recovered: true,
            ..
        }))
    ));
    for records in [[2, 4, 1], [4, 2, 1], [3, 1, 4]] {
        let expected = u64::from(records[0] != 4);
        assert!(
            matches!(scan(policy, &records), Err(PcuVulkanError::Fault(PcuExecutionFault { invocation_id, recovered: false, .. })) if invocation_id == expected)
        );
    }
    assert!(matches!(
        scan(policy, &[2, 4, 5]),
        Err(PcuVulkanError::InvalidStatus {
            invocation_id: 2,
            ..
        })
    ));
}
#[test]
fn unsigned_division_cannot_report_signed_overflow() {
    for scalar in [
        PcuScalarType::U8,
        PcuScalarType::U16,
        PcuScalarType::U32,
        PcuScalarType::U64,
        PcuScalarType::U128,
        PcuScalarType::U256,
        PcuScalarType::U512,
    ] {
        let policy = StatusPolicy::checked(
            PcuCheckedScalarFaultLaw::integer_div_rem(scalar),
            PcuRangePolicy::Reject,
        )
        .unwrap();
        assert!(matches!(scan(policy, &[4]), Err(PcuVulkanError::Fault(_))));
        assert!(matches!(
            scan(policy, &[5]),
            Err(PcuVulkanError::InvalidStatus { .. })
        ));
    }
}
#[test]
fn constructor_retains_operation_and_underflow_policy() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F32,
        PcuScalarType::F64,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ] {
        for underflow in [
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            let policy = StatusPolicy::checked(
                PcuCheckedScalarFaultLaw::float_binary(
                    scalar,
                    PcuDispatchFloatBinaryOp::Add,
                    PcuRangePolicy::Reject,
                    underflow,
                ),
                PcuRangePolicy::Reject,
            )
            .unwrap();
            assert!(matches!(
                scan(policy, &[4]),
                Err(PcuVulkanError::InvalidStatus { .. })
            ));
            if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                assert!(matches!(scan(policy, &[2]), Err(PcuVulkanError::Fault(_))));
            } else {
                assert!(matches!(
                    scan(policy, &[2]),
                    Err(PcuVulkanError::InvalidStatus { .. })
                ));
            }
        }
    }
}

#[test]
fn signed_division_and_integer_ranges_keep_distinct_physical_codes() {
    for scalar in [
        PcuScalarType::I8,
        PcuScalarType::I16,
        PcuScalarType::I32,
        PcuScalarType::I64,
        PcuScalarType::I128,
        PcuScalarType::I256,
        PcuScalarType::I512,
    ] {
        let division = StatusPolicy::checked(
            PcuCheckedScalarFaultLaw::integer_div_rem(scalar),
            PcuRangePolicy::Reject,
        )
        .unwrap();
        for code in [4, 5] {
            assert!(matches!(
                scan(division, &[0, code]),
                Err(PcuVulkanError::Fault(PcuExecutionFault {
                    invocation_id: 1,
                    recovered: false,
                    ..
                }))
            ));
        }
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let add = integer(scalar, PcuDispatchIntegerBinaryOp::Add, range);
            for code in [2, 3] {
                if range == PcuRangePolicy::Clamp {
                    assert!(matches!(
                        scan(add, &[code]),
                        Ok(Some(PcuExecutionFault {
                            recovered: true,
                            ..
                        }))
                    ));
                } else {
                    assert!(matches!(
                        scan(add, &[code]),
                        Err(PcuVulkanError::Fault(PcuExecutionFault {
                            recovered: false,
                            ..
                        }))
                    ));
                }
            }
            assert!(matches!(
                scan(add, &[5]),
                Err(PcuVulkanError::InvalidStatus { .. })
            ));
        }
    }
}

#[test]
fn a_read_failure_after_a_valid_fatal_prevents_publication() {
    let policy = integer(
        PcuScalarType::I512,
        PcuDispatchIntegerBinaryOp::Mul,
        PcuRangePolicy::Reject,
    );
    assert!(matches!(
        policy.scan(2, |index| if index == 0 {
            Ok(3)
        } else {
            Err(PcuVulkanError::Quarantined)
        }),
        Err(PcuVulkanError::Quarantined)
    ));
}

#[cfg(feature = "tensor")]
#[test]
#[ignore = "requires an actual Vulkan device; deterministic mapped-status protocol exercise, not a driver failure"]
fn native_coherent_records_reject_impossible_classes_before_arbitration() {
    #[rustfmt::skip]
    use crate::ffi::{
        VulkanDevice,
        VulkanOwnedBuffer,
    };
    let device = std::rc::Rc::new(VulkanDevice::new().unwrap());
    let mut status = VulkanOwnedBuffer::new(&device, 12).unwrap();
    let policy = integer(
        PcuScalarType::U512,
        PcuDispatchIntegerBinaryOp::Add,
        PcuRangePolicy::Reject,
    );
    for records in [[3_u32, 0, 4], [3, 0, 2], [3, 0, u32::MAX]] {
        let mut bytes = [0; 12];
        for (slot, code) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(records) {
            slot.copy_from_slice(&code.to_ne_bytes());
        }
        // No asynchronous submission uses this actual coherent owner. The synthetic physical
        // records exercise the production reader/decoder; they do not simulate driver failure.
        status.write(&bytes).unwrap();
        assert!(matches!(
            policy.scan(3, |index| status.status(index)),
            Err(PcuVulkanError::InvalidStatus {
                invocation_id: 2,
                ..
            })
        ));
    }
}

//! Refusal is part of the bounded integer compiler contract, before any publication.
#[rustfmt::skip]
use super::{
    integer_ops,with_map,lower_composed_integer_to_spirv,validate_composed_integer_map,
    PcuDispatchDataOp,PcuDispatchIndex,PcuDispatchKernelIr,PcuDispatchOp,
    PcuDispatchValueId,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuReproducibility,
    PcuScalarType,PcuSpirvLoweringOptions,PcuValueType,
};
#[rustfmt::skip]
use fusion_pcu::model::PcuIntegerDivFlags;
use fusion_pcu::{PcuBindingAccess, PcuBindingRef, PcuDispatchAluOp};

fn refused(kernel: &PcuDispatchKernelIr<'_>) {
    assert!(validate_composed_integer_map(kernel).is_err());
    let mut sink = std::vec![0xfeed_beef];
    assert!(
        lower_composed_integer_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink
        )
        .is_err()
    );
    assert_eq!(sink, [0xfeed_beef]);
}

#[test]
#[allow(clippy::too_many_lines)] // Every mutation begins with the same separately admitted exact integer profile.
fn integer_composition_refuses_unqualified_forms_transactionally() {
    for scalar in [
        PcuScalarType::U8,
        PcuScalarType::I8,
        PcuScalarType::U16,
        PcuScalarType::I16,
        PcuScalarType::U32,
        PcuScalarType::I32,
        PcuScalarType::U64,
        PcuScalarType::I64,
        PcuScalarType::U128,
        PcuScalarType::I128,
        PcuScalarType::U256,
        PcuScalarType::I256,
        PcuScalarType::U512,
        PcuScalarType::I512,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            with_map(
                scalar,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                range,
                |kernel| {
                    let original = integer_ops(kernel);
                    for repro in [
                        PcuReproducibility::Unspecified,
                        PcuReproducibility::PortableV1,
                    ] {
                        let mut candidate = *kernel;
                        candidate.ops = &original;
                        candidate
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = repro;
                        assert!(validate_composed_integer_map(&candidate).is_ok());
                        let mut mismatch = original.clone();
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                            range_policy,
                            ..
                        }) = &mut mismatch[2]
                        {
                            *range_policy = if range == PcuRangePolicy::Reject {
                                PcuRangePolicy::Clamp
                            } else {
                                PcuRangePolicy::Reject
                            };
                        }
                        refused(&PcuDispatchKernelIr {
                            ops: &mismatch,
                            ..candidate
                        });
                        let mut foreign = original.clone();
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                            value_type,
                            ..
                        }) = &mut foreign[2]
                        {
                            *value_type = PcuValueType::Scalar(PcuScalarType::F32);
                        }
                        refused(&PcuDispatchKernelIr {
                            ops: &foreign,
                            ..candidate
                        });
                        let mut oversized_ssa = original.clone();
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            result, ..
                        }) = &mut oversized_ssa[0]
                        {
                            *result = PcuDispatchValueId(256);
                        }
                        refused(&PcuDispatchKernelIr {
                            ops: &oversized_ssa,
                            ..candidate
                        });
                        let mut duplicate = original.clone();
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                            result,
                            ..
                        }) = &mut duplicate[2]
                        {
                            *result = PcuDispatchValueId(7);
                        }
                        refused(&PcuDispatchKernelIr {
                            ops: &duplicate,
                            ..candidate
                        });
                        let mut wrapping = original.clone();
                        wrapping[2] = PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                            value_type: PcuValueType::Scalar(scalar),
                            result: PcuDispatchValueId(17),
                            lhs: PcuDispatchValueId(7),
                            rhs: PcuDispatchValueId(13),
                            op: PcuDispatchAluOp::Add,
                        });
                        refused(&PcuDispatchKernelIr {
                            ops: &wrapping,
                            ..candidate
                        });
                        let mut division = original.clone();
                        division[2] = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                            value_type: PcuValueType::Scalar(scalar),
                            flags: PcuIntegerDivFlags::CHECKED,
                            quotient: PcuDispatchValueId(17),
                            remainder: PcuDispatchValueId(18),
                            lhs: PcuDispatchValueId(7),
                            rhs: PcuDispatchValueId(13),
                        });
                        refused(&PcuDispatchKernelIr {
                            ops: &division,
                            ..candidate
                        });
                        let nested = [PcuDispatchOp::GridStrideLoop {
                            extent: 65,
                            body: &original[..6],
                        }];
                        let outer = [PcuDispatchOp::GridStrideLoop {
                            extent: 65,
                            body: &nested,
                        }];
                        refused(&PcuDispatchKernelIr {
                            ops: &outer,
                            ..candidate
                        });
                        let mut empty = candidate;
                        empty.entry.logical_shape = [0, 1, 1];
                        refused(&empty);
                        let mut too_many = original[..6].to_vec();
                        for id in 50..120 {
                            too_many.push(PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                                result: PcuDispatchValueId(id),
                                binding: PcuBindingRef::new(0, 1),
                                index: PcuDispatchIndex::InvocationId,
                            }));
                        }
                        refused(&PcuDispatchKernelIr {
                            ops: &too_many,
                            ..candidate
                        });
                        let mut bindings = kernel.bindings.to_vec();
                        bindings[1].access = PcuBindingAccess::ReadWrite;
                        let mut cross_index = original.clone();
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                            index, ..
                        }) = &mut cross_index[0]
                        {
                            *index = PcuDispatchIndex::BindingElementZero;
                        }
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                            binding,
                            ..
                        }) = &mut cross_index[5]
                        {
                            *binding = PcuBindingRef::new(0, 1);
                        }
                        refused(&PcuDispatchKernelIr {
                            ops: &cross_index,
                            bindings: &bindings,
                            ..candidate
                        });
                    }
                },
            );
        }
    }
}

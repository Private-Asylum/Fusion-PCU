#[path = "fixture/fixture.rs"]
mod fixture;
#[path = "mixed/mixed.rs"]
mod mixed;
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuDispatchKernelIr,
};

#[test]
fn all_twenty_two_raw_carriers_preserve_full_request_and_ordered_roles() {
    for scalar in PcuScalarType::ALL {
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            continue;
        }
        for grid in [false, true] {
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for compound in [
                    PcuCompoundArithmeticPolicy::Checked,
                    PcuCompoundArithmeticPolicy::BackendDefined,
                ] {
                    for precision in [
                        PcuPrecisionPolicy::Preserve,
                        PcuPrecisionPolicy::BackendOptimized,
                    ] {
                        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                            for uf in [
                                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                            ] {
                                let mut requirements = PcuDispatchKernelIr::DEFAULT_REQUIREMENTS;
                                requirements.numerical_mode = mode;
                                requirements.range_policy = range;
                                requirements.float_underflow = uf;
                                requirements.numerical_options.compound_arithmetic = compound;
                                requirements.numerical_options.precision = precision;
                                fixture::visit(scalar, grid, requirements, |kernel| {
                                    let plan = MetalTransportPlan::assess(kernel, scalar).unwrap();
                                    assert_eq!(plan.requirements(), requirements);
                                    assert_eq!(
                                        plan.resources()
                                            .iter()
                                            .map(|role| role.binding)
                                            .collect::<Vec<_>>(),
                                        [
                                            fixture::INPUT,
                                            fixture::STAGE,
                                            fixture::SEED,
                                            fixture::OUTPUT
                                        ]
                                    );
                                    assert_eq!(plan.resources()[1].minimum_read_elements, 17);
                                    assert_eq!(
                                        plan.resources()[1].minimum_initial_read_elements,
                                        0
                                    );
                                    assert_eq!(
                                        plan.resources()[2].minimum_initial_read_elements,
                                        1
                                    );
                                    assert_eq!(plan.element_count(), 17);
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn packed_carriers_and_unproved_portable_request_refuse_before_discovery() {
    for scalar in [PcuScalarType::Bool, PcuScalarType::I4, PcuScalarType::U4] {
        fixture::visit(
            scalar,
            false,
            PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            |kernel| {
                assert!(matches!(
                    MetalTransportPlan::assess(kernel, scalar),
                    Err(MetalError::Unsupported)
                ));
            },
        );
    }
    let mut request = PcuDispatchKernelIr::DEFAULT_REQUIREMENTS;
    request.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    fixture::visit(PcuScalarType::F32, false, request, |kernel| {
        assert!(matches!(
            MetalTransportPlan::assess(kernel, PcuScalarType::F32),
            Err(MetalError::Unsupported)
        ));
    });
}

#[test]
#[ignore = "Requires actual Metal compiler/device; exercises raw encodings and full-capacity writable views."]
fn native_all_twenty_two_transport_encodings_saved_ssa_tails_and_aliases() {
    let session = MetalSession::open(0).unwrap();
    for scalar in PcuScalarType::ALL {
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            continue;
        }
        for grid in [false, true] {
            fixture::visit(
                scalar,
                grid,
                PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                |ir| {
                    let plan = MetalTransportPlan::assess(ir, scalar).unwrap();
                    let width = plan.width;
                    let prepared = session.prepare_transport_plan(plan).unwrap();
                    for phase in 0..3_u8 {
                        let bytes: Vec<_> = (0..23 * width)
                            .map(|index| {
                                u8::try_from(index % 256)
                                    .unwrap()
                                    .wrapping_mul(37)
                                    .wrapping_add(phase)
                            })
                            .collect();
                        let input = session.upload_bytes(&bytes).unwrap();
                        let seed_bytes = vec![phase.wrapping_add(0x83); width];
                        let seed = session.upload_bytes(&seed_bytes).unwrap();
                        let stage_bytes = vec![0x5e; 19 * width];
                        let output_bytes = vec![0xa7; 21 * width];
                        let stage = session.upload_bytes(&stage_bytes).unwrap();
                        let output = session.upload_bytes(&output_bytes).unwrap();
                        prepared
                            .execute_into(&[&input, &stage, &seed, &output])
                            .unwrap();
                        let mut actual_stage = stage_bytes.clone();
                        stage.read_into_bytes(&mut actual_stage).unwrap();
                        let mut expected_stage = seed_bytes.repeat(17);
                        expected_stage.extend_from_slice(&stage_bytes[17 * width..]);
                        assert_eq!(actual_stage, expected_stage);
                        let mut actual = output_bytes.clone();
                        output.read_into_bytes(&mut actual).unwrap();
                        let mut expected = bytes[..17 * width].to_vec();
                        expected.extend_from_slice(&output_bytes[17 * width..]);
                        assert_eq!(actual, expected);
                        let mut incoming = vec![0; bytes.len()];
                        input.read_into_bytes(&mut incoming).unwrap();
                        assert_eq!(incoming, bytes);
                        assert_eq!(
                            prepared.execute_into(&[&input, &stage, &seed, &stage]),
                            Err(MetalError::Unsupported)
                        );
                        let short = session.allocate_zeroed_bytes(16 * width).unwrap();
                        assert_eq!(
                            prepared.execute_into(&[&input, &stage, &seed, &short]),
                            Err(MetalError::InvalidExtent)
                        );
                        let mut retained = vec![0; actual_stage.len()];
                        stage.read_into_bytes(&mut retained).unwrap();
                        assert_eq!(retained, expected_stage);
                    }
                },
            );
        }
    }
}

#[test]
fn cross_lane_mutable_zero_read_refuses_before_native_submission() {
    fixture::visit(
        PcuScalarType::U64,
        false,
        PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        |kernel| {
            let mut ops = kernel.ops.to_vec();
            for op in &mut ops {
                if let fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::BindingLoad {
                    binding,
                    index,
                    ..
                }) = op
                    && *binding == fixture::STAGE
                {
                    *index = fusion_pcu::PcuDispatchIndex::BindingElementZero;
                }
            }
            let changed = PcuDispatchKernelIr {
                ops: &ops,
                ..*kernel
            };
            assert!(matches!(
                MetalTransportPlan::assess(&changed, PcuScalarType::U64),
                Err(MetalError::Unsupported)
            ));
        },
    );
}

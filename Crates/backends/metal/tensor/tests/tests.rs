//! Exact leaf requests, provenance/shape rejection and retained native storage ownership.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{PcuImplementationRequirements,PcuNumericalMode,PcuNumericalOptions,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuReproducibility};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{Graph,TensorArithmeticRewritePolicy,TensorArithmeticCapability,TensorPointwiseGroupingPolicy,ValueId,TensorOwnedSelectedProgram};
fn selected(graph: Graph, output: ValueId) -> TensorOwnedSelectedProgram {
    graph
        .into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap()
}
fn plan(scalar: PcuScalarType, requirements: PcuImplementationRequirements) -> MetalTensorPlan {
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input([5, 13], scalar).unwrap();
    MetalTensorPlan::assess_program(&selected(graph, input), requirements).unwrap()
}
#[test]
fn exact22_leaf_freezes_full_tuple_and_rejects_effects_shapes_and_packed_types() {
    for scalar in PcuScalarType::ALL {
        let mut graph = Graph::try_new().unwrap();
        let input = graph.input([65], scalar).unwrap();
        if scalar.bit_width() < 8 {
            assert!(
                graph
                    .into_selected_program(
                        &[input],
                        TensorArithmeticRewritePolicy::Disabled,
                        TensorArithmeticCapability::Strict,
                        TensorPointwiseGroupingPolicy::Disabled
                    )
                    .is_err()
            );
            continue;
        }
        let program = selected(graph, input);
        assert!(
            MetalTensorPlan::assess_program(&program, PcuImplementationRequirements::default())
                .is_ok()
        );
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            for compound in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    for underflow in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    ] {
                        let requirements = PcuImplementationRequirements {
                            numerical_mode: mode,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic: compound,
                                precision,
                                reproducibility: PcuReproducibility::Unspecified,
                            },
                            float_underflow: underflow,
                            range_policy: PcuRangePolicy::Reject,
                        };
                        let plan = plan(scalar, requirements);
                        assert_eq!(plan.scalar_type(), scalar);
                        assert_eq!(plan.shape(), [5, 13]);
                        assert_eq!(plan.element_count(), 65);
                        assert_eq!(plan.requirements(), requirements);
                        assert!(Rc::ptr_eq(&plan.shape_owner(), &plan.shape_owner()));
                    }
                }
            }
        }
        let mut portable = PcuImplementationRequirements::default();
        portable.numerical_options.reproducibility = PcuReproducibility::PortableV1;
        assert!(MetalTensorPlan::assess_program(&program, portable).is_err());
        let clamp = PcuImplementationRequirements {
            range_policy: PcuRangePolicy::Clamp,
            ..PcuImplementationRequirements::default()
        };
        assert!(MetalTensorPlan::assess_program(&program, clamp).is_err());
    }
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input([3], PcuScalarType::F32).unwrap();
    let _unused = graph.relu(input).unwrap();
    assert!(
        MetalTensorPlan::assess_program(
            &selected(graph, input),
            PcuImplementationRequirements::default()
        )
        .is_err()
    );
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input([3], PcuScalarType::U32).unwrap();
    let output = graph.add(input, input).unwrap();
    assert!(
        MetalTensorPlan::assess_program(
            &selected(graph, output),
            PcuImplementationRequirements::default()
        )
        .is_err()
    );
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input([0], PcuScalarType::U32).unwrap();
    assert!(
        MetalTensorPlan::assess_program(
            &selected(graph, input),
            PcuImplementationRequirements::default()
        )
        .is_err()
    );
}
#[test]
#[ignore = "Requires actual Metal22 leaf fresh ownership, logical payload, same-session composition and foreign rejection."]
fn exact22_native_leaf_ownership_liveness_and_type_preflight() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    for scalar in PcuScalarType::ALL {
        if scalar.bit_width() < 8 {
            continue;
        }
        for phase in 0..3 {
            native_leaf_phase(&session, &foreign, scalar, phase);
        }
    }
}
// Keep each physical ownership boundary together: upload, copy, failed preflights,
// escaped-owner readback and final transfer of the original retained session.
#[allow(clippy::too_many_lines)]
fn native_leaf_phase(
    session: &MetalSession,
    foreign: &MetalSession,
    scalar: PcuScalarType,
    phase: usize,
) {
    let width = usize::from(scalar.bit_width()) / 8;
    let prepared = session
        .prepare_tensor_program(
            plan(scalar, PcuImplementationRequirements::default()),
            PcuMemoryPoolId(129),
        )
        .unwrap();
    let other = foreign
        .prepare_tensor_program(
            plan(scalar, PcuImplementationRequirements::default()),
            PcuMemoryPoolId(129),
        )
        .unwrap();
    let mut bytes = (0..65 * width)
        .map(|index| match phase {
            0 => 0,
            1 => 255,
            _ => u8::try_from((index * 73 + index / width * 19) % 256).unwrap(),
        })
        .collect::<Vec<_>>();
    let expected = bytes.clone();
    let first = prepared
        .execute(MetalTensorInput::HostBytes {
            scalar,
            elements: 65,
            bytes: &bytes,
        })
        .unwrap();
    let mut actual = vec![99_u8; bytes.len() + 3];
    first.read_bytes_into(&mut actual).unwrap();
    bytes.fill(37); // RAM mutation after escape cannot change the retained GPU owner.
    assert_eq!(&actual[..bytes.len()], expected);
    assert_eq!(&actual[bytes.len()..], [99; 3]);
    let second = prepared
        .execute(MetalTensorInput::Resident {
            scalar,
            elements: 65,
            resource: first.resource(),
        })
        .unwrap();
    assert!(
        other
            .execute(MetalTensorInput::Resident {
                scalar,
                elements: 65,
                resource: first.resource()
            })
            .is_err()
    );
    assert!(
        prepared
            .execute(MetalTensorInput::HostBytes {
                scalar,
                elements: 64,
                bytes: &bytes
            })
            .is_err()
    );
    assert!(
        prepared
            .execute(MetalTensorInput::HostBytes {
                scalar: PcuScalarType::Bool,
                elements: 65,
                bytes: &bytes
            })
            .is_err()
    );
    let mut short = vec![77_u8; bytes.len() - 1];
    assert!(second.read_bytes_into(&mut short).is_err());
    assert!(short.iter().all(|&byte| byte == 77));
    if scalar != PcuScalarType::U8 {
        let mut wrong_typed = vec![77_u8; 65 * width];
        assert!(second.read_into(&mut wrong_typed).is_err());
        assert!(wrong_typed.iter().all(|&byte| byte == 77));
    }
    let shape = second.shape_owner();
    assert!(Rc::ptr_eq(&shape, &prepared.plan().shape_owner()));
    drop(first);
    drop(prepared);
    second.read_bytes_into(&mut actual).unwrap();
    assert_eq!(&actual[..bytes.len()], expected);
    let (original, resource) = second.into_resource();
    assert!(session.same_session(&original));
    resource.validate_access_available().unwrap();
    drop(resource);
    drop(original);
}

#[path = "relu/relu.rs"]
mod relu;

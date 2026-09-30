//! Pure admission, ABI and event-location tests; GPU witnesses are separately ignored.
use super::*;
use fusion_pcu::PcuExecutionFaultKind;
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};

fn graph(scalar: PcuScalarType) -> (Graph, ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([2, 3], scalar).unwrap();
    let right = graph.input([3, 4], scalar).unwrap();
    let output = graph.matmul(left, right).unwrap();
    (graph, output)
}

#[test]
fn generated_strict_source_preserves_order_abi_width_and_policy() {
    for (scalar, prefix, width) in [
        (PcuScalarType::F32, "f32", 4),
        (PcuScalarType::F64, "f64", 8),
    ] {
        let (mut graph, output) = graph(scalar);
        for (policy, tag) in [
            (PcuFloatUnderflowPolicy::IeeeAfterRounding, 0),
            (PcuFloatUnderflowPolicy::RejectSubnormalResult, 1),
            (PcuFloatUnderflowPolicy::AllowGradualUnderflow, 2),
        ] {
            graph
                .set_value_float_underflow_policy(output, policy)
                .unwrap();
            let profile = Profile::from_node(&graph, graph.node(output).unwrap()).unwrap();
            let source = lower_strict_matmul_to_cuda_source(&graph, output).unwrap();
            assert!(source.contains("#pragma clang fp contract(off)"));
            assert!(source.contains("if (id >= 8ull) return;"));
            assert!(source.contains("k < 3ull; ++k"));
            assert!(source.contains(&format!("fusion_checked_{prefix}_binary")));
            assert!(source.contains(&format!("2u, {tag}u)")));
            assert!(source.contains(&format!("0u, {tag}u)")));
            assert!(source.find("product =").unwrap() < source.find("sum =").unwrap());
            assert!(source.contains("((event + 1ull) << 3u)"));
            assert!(!source.contains("fmaf("));
            assert!(!source.contains("__fma"));
            assert!(!source.contains("__builtin_clz"));
            assert!(source.contains("__clz"));
            assert_eq!(source, profile.source());
            let requirements = profile.requirements();
            assert_eq!(
                requirements
                    .iter()
                    .map(|entry| entry.min_required_bytes)
                    .collect::<Vec<_>>(),
                [6 * width, 12 * width, 8 * width]
            );
            assert_eq!(requirements[2].access, PcuBindingAccess::WriteOnly);
        }
    }
}

#[test]
fn profile_policy_and_shape_are_full_cache_identity_and_boundary_is_rejected() {
    let (mut graph, output) = graph(PcuScalarType::F32);
    let ieee = Profile::from_node(&graph, graph.node(output).unwrap()).unwrap();
    graph
        .set_value_float_underflow_policy(output, PcuFloatUnderflowPolicy::AllowGradualUnderflow)
        .unwrap();
    let gradual = Profile::from_node(&graph, graph.node(output).unwrap()).unwrap();
    assert_ne!(ieee, gradual);
    assert_ne!(
        super::super::TensorDispatchCacheKey::StrictMatMul(ieee),
        super::super::TensorDispatchCacheKey::StrictMatMul(gradual)
    );
    graph
        .set_value_numerical_mode(output, PcuNumericalMode::Boundary)
        .unwrap();
    assert!(lower_strict_matmul_to_cuda_source(&graph, output).is_err());
    let (graph, output) = graph_f64();
    let wide = Profile::from_node(&graph, graph.node(output).unwrap()).unwrap();
    assert_ne!(ieee, wide);
}

fn graph_f64() -> (Graph, ValueId) {
    graph(PcuScalarType::F64)
}

#[test]
fn compound_status_decodes_exact_cell_k_step_and_rejects_invalid_events() {
    let (graph, output) = graph(PcuScalarType::F32);
    let profile = Profile::from_node(&graph, graph.node(output).unwrap()).unwrap();
    for (event, cell, k, step) in [
        (0, 0, 0, TensorArithmeticStep::Multiply),
        (1, 0, 0, TensorArithmeticStep::Add),
        (47, 7, 2, TensorArithmeticStep::Add),
    ] {
        assert!(matches!(profile.fault(output, PcuExecutionFault {
            invocation_id: event, kind: PcuExecutionFaultKind::ArithmeticOverflow, recovered: false,
        }).unwrap(), TensorError::CompoundArithmeticFault {
            value, element_index, reduction_index, step: actual_step,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
        } if value == output && element_index == cell && reduction_index == k && actual_step == step));
    }
    assert!(
        profile
            .fault(
                output,
                PcuExecutionFault {
                    invocation_id: 48,
                    kind: PcuExecutionFaultKind::ArithmeticOverflow,
                    recovered: false,
                }
            )
            .is_err()
    );
    assert!(
        profile
            .fault(
                output,
                PcuExecutionFault {
                    invocation_id: 0,
                    kind: PcuExecutionFaultKind::ArithmeticOverflow,
                    recovered: true,
                }
            )
            .is_err()
    );
}

#[test]
fn admission_rejects_empty_extents_and_fault_encoding_overflow_before_execution() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([1, 0], PcuScalarType::F32).unwrap();
    let right = graph.input([0, 1], PcuScalarType::F32).unwrap();
    let output = graph.matmul(left, right).unwrap();
    assert!(lower_strict_matmul_to_cuda_source(&graph, output).is_err());
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph
        .input(
            [
                usize::try_from(u32::MAX).unwrap(),
                usize::try_from(u32::MAX).unwrap(),
            ],
            PcuScalarType::F32,
        )
        .unwrap();
    let right = graph
        .input([usize::try_from(u32::MAX).unwrap(), 1], PcuScalarType::F32)
        .unwrap();
    let output = graph.matmul(left, right).unwrap();
    assert!(lower_strict_matmul_to_cuda_source(&graph, output).is_err());
}

#[path = "hardware.rs"]
mod hardware;

#[test]
fn selected_strict_preparation_and_prewarm_reuse_dispatch_spine_without_blas() {
    let (graph, output) = graph(PcuScalarType::F32);
    let prepared = super::super::tests::prepared_for_request_test(
        &graph,
        &[output],
        TensorPointwiseGroupingPolicy::Disabled,
    );
    assert!(!prepared.data.requires_blas);
    let requests = super::super::collect_dispatch_requests(&prepared).unwrap();
    assert_eq!(requests.len(), 1);
    let profile = Profile::from_node(&graph, graph.node(output).unwrap()).unwrap();
    assert_eq!(
        requests[0].key(),
        super::super::TensorDispatchCacheKey::StrictMatMul(profile)
    );
    assert_eq!(
        prepared.data.matmul_operands[2].unwrap().strict_profile,
        Some(profile)
    );
}

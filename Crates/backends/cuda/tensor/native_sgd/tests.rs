//! Native optimizer admission, rounding and complete-output ownership witnesses.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuCompoundArithmeticPolicy,
    PcuDeviceTensor,
    PcuFloatUnderflowPolicy,
    PcuMemoryPoolId,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuNumericalRequirement,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    OpDescriptor,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorError,
    TensorPointwiseGroupingPolicy,
    TensorUnsupportedReason,
};
use super::super::CudaTensorAssessor;

fn options(precision: PcuPrecisionPolicy) -> PcuNumericalOptions {
    PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision,
        reproducibility: PcuReproducibility::Unspecified,
    }
}

#[test]
fn native_sgd_policy_and_finite_frozen_rate_are_admitted_cold() {
    let mut graph = Graph::default();
    let weights = graph.input([17], PcuScalarType::F32).unwrap();
    let gradient = graph.input([17], PcuScalarType::F32).unwrap();
    for rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            graph.sgd_update(weights, gradient, rate),
            Err(TensorError::InvalidLearningRate)
        );
    }
    let updated = graph.sgd_update(weights, gradient, -0.5).unwrap();
    let mut node = graph.node(updated).unwrap();
    assert_eq!(
        super::assess(node),
        Err(TensorUnsupportedReason::NumericalPolicy {
            requirement: PcuNumericalRequirement::CompoundArithmetic,
            options: PcuNumericalOptions::default(),
        })
    );
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        node.numerical_options = options(precision);
        assert!(super::assess(node).is_ok());
        for mode in [None, Some(PcuNumericalMode::Strict)] {
            node.numerical_mode = mode;
            assert!(matches!(
                super::assess(node),
                Err(TensorUnsupportedReason::NumericalPolicy {
                    requirement: PcuNumericalRequirement::CompoundArithmetic,
                    ..
                })
            ));
        }
        node.numerical_mode = Some(PcuNumericalMode::Boundary);
        node.numerical_options.reproducibility = PcuReproducibility::PortableV1;
        assert!(matches!(
            super::assess(node),
            Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::Reproducibility,
                ..
            })
        ));
        node.numerical_options = options(precision);
        node.float_underflow_policy = Some(PcuFloatUnderflowPolicy::RejectSubnormalResult);
        assert_eq!(
            super::assess(node),
            Err(TensorUnsupportedReason::UnderflowPolicy(
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ))
        );
        node.float_underflow_policy = None;
        node.scalar_type = PcuScalarType::F64;
        assert_eq!(
            super::assess(node),
            Err(TensorUnsupportedReason::ElementType)
        );
        node.scalar_type = PcuScalarType::F32;
    }
}

#[test]
fn native_sgd_rejects_forged_rate_bits_and_scalar_rewrite_provenance() {
    let mut graph = Graph::default();
    graph.set_numerical_options(options(PcuPrecisionPolicy::Preserve));
    let weights = graph.input([1], PcuScalarType::F32).unwrap();
    let gradient = graph.input([1], PcuScalarType::F32).unwrap();
    let updated = graph.sgd_update(weights, gradient, 0.0).unwrap();
    let mut node = graph.node(updated).unwrap();
    assert!(super::assess_graph(&graph, node).is_ok());
    node.op = OpDescriptor::SgdUpdate {
        weights,
        gradient,
        learning_rate: -0.0,
    };
    assert_eq!(
        super::assess_graph(&graph, node),
        Err(TensorUnsupportedReason::Operation)
    );

    let scaled = graph.mul(weights, gradient).unwrap();
    let ordinary = graph.sub(weights, scaled).unwrap();
    let mut rewritten = graph.node(ordinary).unwrap();
    rewritten.op = OpDescriptor::SgdUpdate {
        weights,
        gradient,
        learning_rate: 0.5,
    };
    assert_eq!(
        super::assess_graph(&graph, rewritten),
        Err(TensorUnsupportedReason::Operation)
    );
}

fn execute(
    session: &crate::CudaOwnedDispatchBackend,
    assessor: &CudaTensorAssessor<'_>,
    weights: &[f32],
    gradient: &[f32],
    rate: f32,
    precision: PcuPrecisionPolicy,
) -> Vec<f32> {
    let mut graph = Graph::default();
    graph.set_numerical_options(options(precision));
    let w = graph.input([weights.len()], PcuScalarType::F32).unwrap();
    let g = graph.input([gradient.len()], PcuScalarType::F32).unwrap();
    let output = graph.sgd_update(w, g, rate).unwrap();
    let prepared = assessor
        .prepare_owned_program(
            graph
                .into_selected_program(
                    &[output],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap(),
        )
        .unwrap();
    let pool = PcuMemoryPoolId(0x5347_4443);
    let w_buffer = PcuDeviceTensor::new(
        [weights.len()],
        session.upload_buffer(pool, weights).unwrap(),
    )
    .unwrap();
    let g_buffer = PcuDeviceTensor::new(
        [gradient.len()],
        session.upload_buffer(pool, gradient).unwrap(),
    )
    .unwrap();
    let mut memory = session.memory_provider(pool);
    let mut actual = vec![0.0_f32; weights.len()];
    for _ in 0..2 {
        let outputs = assessor
            .execute_owned_program_outputs(
                &prepared,
                &[(w, &w_buffer), (g, &g_buffer)],
                pool,
                &mut memory,
            )
            .unwrap();
        assert_eq!(outputs[0].1.shape(), [weights.len()]);
        session
            .download_buffer(pool, outputs[0].1.buffer(), &mut actual)
            .unwrap();
    }
    actual
}

#[test]
#[ignore = "requires an idle CUDA device and CUDA Toolkit"]
fn native_sgd_complete_output_frozen_rates_rounding_and_exception_permission() {
    let (_discovery, session) = super::super::tests::cuda_test_session();
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let weights: [f32; 17] = core::array::from_fn(|index| f32::from(u16::try_from(index).unwrap()));
    let gradient: [f32; 17] =
        core::array::from_fn(|index| f32::from(i16::try_from(index % 3).unwrap() - 1));
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        for rate in [-0.0_f32, 0.0, -0.5, 0.5, 1.0] {
            let expected: Vec<_> = weights
                .iter()
                .zip(gradient)
                .map(|(&weight, gradient)| {
                    if precision == PcuPrecisionPolicy::BackendOptimized {
                        (-rate).mul_add(gradient, weight)
                    } else {
                        weight
                            .pcu_checked_sub(rate.pcu_checked_mul(gradient).unwrap())
                            .unwrap()
                    }
                })
                .map(f32::to_bits)
                .collect();
            assert_eq!(
                execute(&session, &assessor, &weights, &gradient, rate, precision)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                expected
            );
        }
        for rate in [f32::MAX, -f32::MAX] {
            assert_eq!(
                execute(
                    &session,
                    &assessor,
                    &[1.0, 2.0],
                    &[0.0, 0.0],
                    rate,
                    precision
                ),
                [1.0, 2.0]
            );
        }
        let nonfinite = execute(
            &session,
            &assessor,
            &[f32::INFINITY, 1.0],
            &[1.0, f32::NAN],
            0.5,
            precision,
        );
        assert!(nonfinite.iter().all(|value| !value.is_finite()));
        // A successful call after the permitted nonfinite result reuses the selected source cache.
        assert_eq!(
            execute(&session, &assessor, &[1.0], &[1.0], 0.5, precision),
            [0.5]
        );
        let rate = f32::from_bits(0x3f80_0001);
        let gradient = f32::from_bits(0x3f7f_fffe);
        let actual = execute(&session, &assessor, &[1.0], &[gradient], rate, precision);
        assert_eq!(
            actual[0].to_bits(),
            if precision == PcuPrecisionPolicy::BackendOptimized {
                (127_u32 - 46) << 23
            } else {
                0
            }
        );
    }
}

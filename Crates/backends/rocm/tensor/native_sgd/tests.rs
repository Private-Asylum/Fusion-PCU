//! Native SGD policy, immutable parameter provenance, rounding, and escaped ownership.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorError,
    TensorPointwiseGroupingPolicy,
    ValueId,
};

fn options(precision: PcuPrecisionPolicy) -> PcuNumericalOptions {
    PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision,
        reproducibility: PcuReproducibility::Unspecified,
    }
}

fn graph(shape: &[usize], scalar: PcuScalarType, rate: f32) -> (Graph, ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_options(options(PcuPrecisionPolicy::Preserve));
    let weights = graph.input(shape.to_vec(), scalar).unwrap();
    let gradient = graph.input(shape.to_vec(), scalar).unwrap();
    let output = graph.sgd_update(weights, gradient, rate).unwrap();
    (graph, output)
}

#[test]
fn policy_matrix_requires_explicit_native_boundary() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for reproducibility in [
                    PcuReproducibility::Unspecified,
                    PcuReproducibility::PortableV1,
                ] {
                    let (mut graph, output) = graph(&[17], PcuScalarType::F32, 0.5);
                    graph.set_value_numerical_mode(output, mode).unwrap();
                    graph
                        .set_value_numerical_options(
                            output,
                            PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                reproducibility,
                            },
                        )
                        .unwrap();
                    assert_eq!(
                        matches!(
                            assess(&graph, graph.node(output).unwrap()),
                            TensorOperationSupport::Supported {
                                route: TensorExecutionRoute::Native,
                                workspace_bytes: Some(0)
                            }
                        ),
                        mode == PcuNumericalMode::Boundary
                            && compound_arithmetic == PcuCompoundArithmeticPolicy::BackendDefined
                            && reproducibility == PcuReproducibility::Unspecified
                    );
                }
            }
        }
    }
}

#[test]
fn immutable_rate_and_policy_provenance_include_signed_zero() {
    let (mut graph, output) = graph(&[17], PcuScalarType::F32, 0.0);
    let mut forged = graph.node(output).unwrap();
    let OpDescriptor::SgdUpdate {
        weights, gradient, ..
    } = forged.op
    else {
        unreachable!()
    };
    for rate in [-0.0, 0.5, f32::NAN, f32::INFINITY] {
        forged.op = OpDescriptor::SgdUpdate {
            weights,
            gradient,
            learning_rate: rate,
        };
        assert_eq!(
            assess(&graph, forged),
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Operation
            }
        );
    }
    for rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            graph.sgd_update(weights, gradient, rate),
            Err(TensorError::InvalidLearningRate)
        );
    }
    forged = graph.node(output).unwrap();
    forged.numerical_options.precision = PcuPrecisionPolicy::BackendOptimized;
    assert_eq!(
        assess(&graph, forged),
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Operation
        }
    );
    graph
        .set_value_float_underflow_policy(output, PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap();
    assert_eq!(
        assess(&graph, graph.node(output).unwrap()),
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::UnderflowPolicy(
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            )
        }
    );
}

#[test]
fn dense_extent_and_scalar_width_are_admitted_cold() {
    assert_eq!(
        Graph::default().input([usize::MAX, 2], PcuScalarType::F32),
        Err(TensorError::ShapeOverflow)
    );
    for shape in [&[0][..], &[u32::MAX as usize, 2][..]] {
        let (graph, output) = graph(shape, PcuScalarType::F32, -0.5);
        assert_eq!(
            assess(&graph, graph.node(output).unwrap()),
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape
            }
        );
    }
    let mut f64_graph = Graph::default();
    let weights = f64_graph.input([17], PcuScalarType::F64).unwrap();
    let gradient = f64_graph.input([17], PcuScalarType::F64).unwrap();
    assert!(matches!(
        f64_graph.sgd_update(weights, gradient, 0.5),
        Err(TensorError::UnsupportedScalarType {
            scalar_type: PcuScalarType::F64,
            ..
        })
    ));
    let (graph, output) = graph(&[17], PcuScalarType::F32, 0.5);
    let mut forged = graph.node(output).unwrap();
    forged.scalar_type = PcuScalarType::F64;
    assert_eq!(
        assess(&graph, forged),
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Operation
        }
    );
}

fn execute(
    session: &crate::RocmOwnedDispatchBackend,
    assessor: &RocmTensorAssessor<'_>,
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
    let pool = PcuMemoryPoolId(0x5347_4452);
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
    // Reject an incomplete binding set before publishing output, then retry the same program.
    assert!(
        assessor
            .execute_owned_program_outputs(&prepared, &[(w, &w_buffer)], pool, &mut memory)
            .is_err()
    );
    let outputs = assessor
        .execute_owned_program_outputs(
            &prepared,
            &[(w, &w_buffer), (g, &g_buffer)],
            pool,
            &mut memory,
        )
        .unwrap();
    // A later fresh-output invocation cannot invalidate this escaped first output.
    let later = assessor
        .execute_owned_program_outputs(
            &prepared,
            &[(w, &w_buffer), (g, &g_buffer)],
            pool,
            &mut memory,
        )
        .unwrap();
    drop(later);
    drop(w_buffer);
    drop(g_buffer);
    drop(prepared);
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].0, output);
    assert_eq!(outputs[0].1.shape(), [weights.len()]);
    let mut actual = vec![0.0; weights.len()];
    session
        .download_buffer(pool, outputs[0].1.buffer(), &mut actual)
        .unwrap();
    actual
}

fn reject_before_kernel_preparation(assessor: &RocmTensorAssessor<'_>) {
    let (mut rejected, update) = graph(&[17], PcuScalarType::F32, 0.5);
    let OpDescriptor::SgdUpdate {
        weights, gradient, ..
    } = rejected.node(update).unwrap().op
    else {
        unreachable!()
    };
    let unsupported = rejected.relu_backward(weights, gradient).unwrap();
    assert!(
        assessor
            .prepare_graph_outputs(&rejected, &[update, unsupported])
            .is_err()
    );
    assert!(
        assessor
            .prepare_owned_program(
                rejected
                    .into_selected_program(
                        &[update, unsupported],
                        TensorArithmeticRewritePolicy::Disabled,
                        TensorArithmeticCapability::Strict,
                        TensorPointwiseGroupingPolicy::Disabled
                    )
                    .unwrap()
            )
            .is_err()
    );
    assert!(assessor.state().native_sgd.preserved.borrow().is_none());
    assert!(assessor.state().native_sgd.contracted.borrow().is_none());
}

#[test]
#[ignore = "requires an idle ROCm device and HIP compiler"]
fn native_sgd_rounding_rates_exception_permission_and_retry() {
    let (_discovery, session) = crate::tensor::tests::rocm_test_session();
    let assessor = RocmTensorAssessor::new(&session).unwrap();
    reject_before_kernel_preparation(&assessor);
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
        let exceptional = execute(
            &session,
            &assessor,
            &[f32::INFINITY, 1.0, f32::MAX],
            &[1.0, f32::NAN, -f32::MAX],
            2.0,
            precision,
        );
        assert!(exceptional.iter().all(|value| !value.is_finite()));
        assert_eq!(
            execute(&session, &assessor, &[1.0], &[1.0], 0.5, precision),
            [0.5]
        );
        let witness = execute(
            &session,
            &assessor,
            &[1.0; 17],
            &[f32::from_bits(0x3f7f_fffe); 17],
            f32::from_bits(0x3f80_0001),
            precision,
        );
        let expected = if precision == PcuPrecisionPolicy::BackendOptimized {
            (127_u32 - 46) << 23
        } else {
            0
        };
        assert!(witness.iter().all(|value| value.to_bits() == expected));
    }
}

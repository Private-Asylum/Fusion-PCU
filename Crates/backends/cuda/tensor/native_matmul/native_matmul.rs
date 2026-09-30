//! Policy admission and real-device acceptance for explicitly native compound matrix products.

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuCompoundArithmeticPolicy,
        PcuDeviceTensor,
        PcuFloatUnderflowPolicy,
        PcuMemoryPoolId,
        PcuNumericalMode,
        PcuNumericalOptions,
        PcuNumericalRequirement,
        PcuPrecisionPolicy,
        PcuReproducibility,
        PcuScalar,
        PcuScalarType,
    };
    #[rustfmt::skip]
    use fusion_pcu::dialect::tensor::{
        Graph,
        TensorArithmeticCapability,
        TensorArithmeticRewritePolicy,
        TensorExecutionRoute,
        TensorOperationAssessor,
        TensorOperationSupport,
        TensorPointwiseGroupingPolicy,
        TensorUnsupportedReason,
        ValueId,
    };
    #[rustfmt::skip]
    use crate::{
        CublasEnvironmentSnapshot,
        CudaOwnedDispatchBackend,
        CudaTensorAssessor,
    };
    #[rustfmt::skip]
    use super::super::{
        assess_matmul_numerical_options,
        nodes_use_native_matmul_batch,
    };

    fn native_options(precision: PcuPrecisionPolicy) -> PcuNumericalOptions {
        PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision,
            reproducibility: PcuReproducibility::Unspecified,
        }
    }

    #[test]
    fn independent_matmul_policy_matrix_rejects_unproved_combinations_cold() {
        let mut graph = Graph::default();
        let left = graph.input([2, 3], PcuScalarType::F64).unwrap();
        let right = graph.input([3, 2], PcuScalarType::F64).unwrap();
        let output = graph.matmul(left, right).unwrap();
        let environment = CublasEnvironmentSnapshot::capture();
        let node = graph.node(output).unwrap();
        assert_eq!(
            assess_matmul_numerical_options(node, &environment),
            Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::CompoundArithmetic,
                options: PcuNumericalOptions::default(),
            })
        );
        graph
            .set_value_numerical_options(
                output,
                native_options(PcuPrecisionPolicy::BackendOptimized),
            )
            .unwrap();
        assert!(assess_matmul_numerical_options(graph.node(output).unwrap(), &environment).is_ok());
        let mut absent_mode = graph.node(output).unwrap();
        absent_mode.numerical_mode = None;
        assert!(matches!(
            assess_matmul_numerical_options(absent_mode, &environment),
            Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::CompoundArithmetic,
                ..
            })
        ));
        graph
            .set_value_numerical_mode(output, PcuNumericalMode::Strict)
            .unwrap();
        assert_eq!(
            assess_matmul_numerical_options(graph.node(output).unwrap(), &environment),
            Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::CompoundArithmetic,
                options: native_options(PcuPrecisionPolicy::BackendOptimized),
            })
        );
        graph
            .set_value_numerical_options(output, PcuNumericalOptions::default())
            .unwrap();
        assert!(assess_matmul_numerical_options(graph.node(output).unwrap(), &environment).is_ok());
        graph
            .set_value_numerical_options(
                output,
                PcuNumericalOptions {
                    reproducibility: PcuReproducibility::PortableV1,
                    ..PcuNumericalOptions::default()
                },
            )
            .unwrap();
        assert_eq!(
            assess_matmul_numerical_options(graph.node(output).unwrap(), &environment),
            Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::Reproducibility,
                options: PcuNumericalOptions {
                    reproducibility: PcuReproducibility::PortableV1,
                    ..PcuNumericalOptions::default()
                },
            })
        );
        graph
            .set_value_numerical_mode(output, PcuNumericalMode::Boundary)
            .unwrap();
        graph
            .set_value_numerical_options(
                output,
                native_options(PcuPrecisionPolicy::BackendOptimized),
            )
            .unwrap();
        graph
            .set_value_float_underflow_policy(
                output,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            )
            .unwrap();
        assert_eq!(
            assess_matmul_numerical_options(graph.node(output).unwrap(), &environment),
            Err(TensorUnsupportedReason::UnderflowPolicy(
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ))
        );
    }

    #[test]
    fn native_event_selection_preserves_other_completion_boundaries() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let right = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let output = graph.matmul(left, right).unwrap();
        let selected = [left, right, output];
        let eligible = |graph: &Graph, values: &[ValueId]| {
            nodes_use_native_matmul_batch(
                &values
                    .iter()
                    .map(|&value| graph.node(value).unwrap())
                    .collect::<Vec<_>>(),
            )
        };
        assert!(!eligible(&graph, &[left, right]));
        assert!(!eligible(&graph, &selected));
        graph
            .set_value_numerical_options(output, native_options(PcuPrecisionPolicy::Preserve))
            .unwrap();
        assert!(eligible(&graph, &selected));
        graph
            .set_value_numerical_mode(output, PcuNumericalMode::Strict)
            .unwrap();
        assert!(!eligible(&graph, &selected));
        graph
            .set_value_numerical_mode(output, PcuNumericalMode::Boundary)
            .unwrap();
        let mixed = graph.add(output, left).unwrap();
        assert!(!eligible(&graph, &[left, right, output, mixed]));
    }

    fn run<T: PcuScalar>(
        session: &CudaOwnedDispatchBackend,
        assessor: &CudaTensorAssessor<'_>,
        left: &[T],
        right: &[T],
        shapes: [[usize; 2]; 2],
        transpose: [bool; 2],
        precision: PcuPrecisionPolicy,
    ) -> Vec<T> {
        let mut graph = Graph::default();
        graph.set_numerical_options(native_options(precision));
        let a = graph.input(shapes[0], T::TYPE).unwrap();
        let b = graph.input(shapes[1], T::TYPE).unwrap();
        let output = graph
            .matmul_transposed(a, b, transpose[0], transpose[1])
            .unwrap();
        assert!(matches!(
            assessor.assess_node(&graph, graph.node(output).unwrap()),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Library,
                ..
            }
        ));
        let count: usize = graph.shape(output).unwrap().iter().product();
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let prepared = assessor.prepare_owned_program(program).unwrap();
        let pool = PcuMemoryPoolId(0x4e41_4355);
        let marker = left[0];
        let left =
            PcuDeviceTensor::new(shapes[0], session.upload_buffer(pool, left).unwrap()).unwrap();
        let right =
            PcuDeviceTensor::new(shapes[1], session.upload_buffer(pool, right).unwrap()).unwrap();
        let mut memory = session.memory_provider(pool);
        let outputs = assessor
            .execute_owned_program_outputs(&prepared, &[(a, &left), (b, &right)], pool, &mut memory)
            .unwrap();
        let mut actual = vec![marker; count];
        session
            .download_buffer(pool, outputs[0].1.buffer(), &mut actual)
            .unwrap();
        actual
    }

    #[test]
    #[ignore = "requires an idle CUDA device and CUDA Toolkit 13.4 cuBLAS component 13.7/13.8"]
    #[allow(clippy::too_many_lines)] // One session verifies mode isolation, transpose and terminal publication together.
    fn native_matmul_precision_transpose_exception_permission_and_cache_separation() {
        let (_discovery, session) = crate::tensor::tests::cuda_test_session();
        let assessor = CudaTensorAssessor::new(&session).unwrap();
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            for transpose in [[false, false], [false, true], [true, false], [true, true]] {
                let left = if transpose[0] {
                    [1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]
                } else {
                    [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                };
                let right = if transpose[1] {
                    [1.0_f32, 3.0, 5.0, 2.0, 4.0, 6.0]
                } else {
                    [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                };
                let shapes = [
                    if transpose[0] { [3, 2] } else { [2, 3] },
                    if transpose[1] { [2, 3] } else { [3, 2] },
                ];
                assert_eq!(
                    run(
                        &session, &assessor, &left, &right, shapes, transpose, precision
                    ),
                    [22.0_f32, 28.0, 49.0, 64.0]
                );
                assert_eq!(
                    run(
                        &session,
                        &assessor,
                        &left.map(f64::from),
                        &right.map(f64::from),
                        shapes,
                        transpose,
                        precision
                    ),
                    [22.0_f64, 28.0, 49.0, 64.0]
                );
            }
        }
        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let precise = [
            f32::from_bits(0x3f80_0001),
            f32::from_bits(0x4000_0001),
            -3.5,
            4.25,
        ];
        assert_eq!(
            run(
                &session,
                &assessor,
                &precise,
                &identity,
                [[2, 2]; 2],
                [false; 2],
                PcuPrecisionPolicy::Preserve
            )
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
            precise.map(f32::to_bits)
        );
        let precise = [
            16_777_217.0_f64,
            f64::from_bits(0x3ff0_0000_0000_0001),
            -3.5,
            4.25,
        ];
        assert_eq!(
            run(
                &session,
                &assessor,
                &precise,
                &identity.map(f64::from),
                [[2, 2]; 2],
                [false; 2],
                PcuPrecisionPolicy::Preserve
            )
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
            precise.map(f64::to_bits)
        );
        // The explicit vendor numerical contract permits nonfinite results, while all API,
        // shape, ownership and terminal completion checks remain in the normal execution path.
        let actual = run(
            &session,
            &assessor,
            &[f32::INFINITY, 2.0, 3.0, 4.0],
            &identity,
            [[2, 2]; 2],
            [false; 2],
            PcuPrecisionPolicy::Preserve,
        );
        assert!(actual.iter().any(|value| !value.is_finite()));
        for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                let handle = assessor.native_cublas(scalar, precision).unwrap();
                let (config, modes) = handle.numerical_config().unwrap();
                assert_eq!(config.scalar_type(), scalar);
                assert_eq!(config.precision(), precision);
                assert_eq!(modes.library_major, 13);
                assert!(matches!(modes.library_minor, 7 | 8));
                assert_eq!(
                    modes.math_mode,
                    if scalar == PcuScalarType::F32
                        && precision == PcuPrecisionPolicy::BackendOptimized
                    {
                        3
                    } else {
                        0
                    }
                );
                assert_eq!(modes.atomics_mode, 1);
                assert!(std::ptr::eq(
                    handle,
                    assessor.native_cublas(scalar, precision).unwrap()
                ));
            }
            assert!(!std::ptr::eq(
                assessor
                    .native_cublas(scalar, PcuPrecisionPolicy::Preserve)
                    .unwrap(),
                assessor
                    .native_cublas(scalar, PcuPrecisionPolicy::BackendOptimized)
                    .unwrap()
            ));
        }
    }

    #[test]
    #[ignore = "requires an idle CUDA device and CUDA Toolkit 13.4 cuBLAS component 13.7/13.8"]
    fn native_matmul_dependency_batch_preserves_selected_output_order_and_handle_reuse() {
        let (_discovery, session) = crate::tensor::tests::cuda_test_session();
        let assessor = CudaTensorAssessor::new(&session).unwrap();
        for precisions in [
            [PcuPrecisionPolicy::Preserve; 2],
            [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ],
            [
                PcuPrecisionPolicy::BackendOptimized,
                PcuPrecisionPolicy::Preserve,
            ],
            [PcuPrecisionPolicy::BackendOptimized; 2],
        ] {
            verify_chain(
                &session,
                &assessor,
                &[1.0_f32, 2.0, 3.0, 4.0],
                &[2.0_f32, 0.0, 0.0, 2.0],
                &[[4.0_f32, 8.0, 12.0, 16.0], [2.0, 4.0, 6.0, 8.0]],
                precisions,
            );
            verify_chain(
                &session,
                &assessor,
                &[1.0_f64, 2.0, 3.0, 4.0],
                &[2.0_f64, 0.0, 0.0, 2.0],
                &[[4.0_f64, 8.0, 12.0, 16.0], [2.0, 4.0, 6.0, 8.0]],
                precisions,
            );
        }
    }

    fn verify_chain<T: PcuScalar + PartialEq + std::fmt::Debug>(
        session: &CudaOwnedDispatchBackend,
        assessor: &CudaTensorAssessor<'_>,
        left: &[T; 4],
        right: &[T; 4],
        expected: &[[T; 4]; 2],
        precisions: [PcuPrecisionPolicy; 2],
    ) {
        let mut graph = Graph::default();
        graph.set_numerical_options(native_options(precisions[0]));
        let a = graph.input([2, 2], T::TYPE).unwrap();
        let b = graph.input([2, 2], T::TYPE).unwrap();
        let first = graph.matmul(a, b).unwrap();
        let second = graph.matmul(first, b).unwrap();
        graph
            .set_value_numerical_options(second, native_options(precisions[1]))
            .unwrap();
        let program = graph
            .into_selected_program(
                &[second, first],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let prepared = assessor.prepare_owned_program(program).unwrap();
        assert!(prepared.data.native_matmul_batch);
        let pool = PcuMemoryPoolId(0x4e41_4355);
        let left =
            PcuDeviceTensor::new([2, 2], session.upload_buffer(pool, left).unwrap()).unwrap();
        let right =
            PcuDeviceTensor::new([2, 2], session.upload_buffer(pool, right).unwrap()).unwrap();
        let mut memory = session.memory_provider(pool);
        // Two calls exercise a reused immutable handle after every terminal event has released
        // all external reservations, including a selected intermediate also read by its child.
        for _ in 0..2 {
            let outputs = assessor
                .execute_owned_program_outputs(
                    &prepared,
                    &[(a, &left), (b, &right)],
                    pool,
                    &mut memory,
                )
                .unwrap();
            assert_eq!(outputs.len(), 2);
            assert_eq!(outputs[0].0, second);
            assert_eq!(outputs[1].0, first);
            for (output, expected) in outputs.iter().zip(expected) {
                let mut actual = [expected[0]; 4];
                session
                    .download_buffer(pool, output.1.buffer(), &mut actual)
                    .unwrap();
                assert_eq!(&actual, expected);
            }
            for precision in precisions {
                assert!(
                    assessor
                        .native_cublas(T::TYPE, precision)
                        .unwrap()
                        .is_usable()
                );
            }
        }
    }
}

#[rustfmt::skip]
use super::{
    MlxError,
    MlxRuntime,
};

#[test]
fn absent_bridge_is_an_explicit_error_without_fallback() {
    let result = MlxRuntime::load("/__fusion_pcu_absent_mlx_bridge__");
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        assert!(matches!(result, Err(MlxError::Unavailable(_))));
    } else {
        assert!(matches!(result, Err(MlxError::UnsupportedPlatform)));
    }
}

#[cfg(feature = "tensor")]
mod gpu {
    #[rustfmt::skip]
    use super::{
        MlxError,
        MlxRuntime,
    };
    #[rustfmt::skip]
    use crate::{
        MlxArrayResidency,
        MlxSession,
    };
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuCompoundArithmeticPolicy,
        PcuNumericalOptions,
        PcuPrecisionPolicy,
        PcuScalarType,
    };
    use fusion_pcu::dialect::tensor::Graph;

    fn runtime() -> MlxRuntime {
        let path = std::env::var_os("PCU_MLX_BRIDGE")
            .expect("set PCU_MLX_BRIDGE to the isolated pinned bridge");
        MlxRuntime::load(path).unwrap()
    }

    fn prepared(session: &MlxSession) -> crate::MlxPreparedMatmul {
        let mut graph = Graph::default();
        graph.set_numerical_options(PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            ..PcuNumericalOptions::default()
        });
        let a = graph.input([2, 3], PcuScalarType::F32).unwrap();
        let b = graph.input([3, 2], PcuScalarType::F32).unwrap();
        let c = graph.matmul(a, b).unwrap();
        session
            .prepare_matmul(&graph, graph.node(c).unwrap())
            .unwrap()
    }

    #[test]
    #[ignore = "requires pinned MLX GPU bridge and an external GPU activity check"]
    fn changing_matrices_terminal_readback_and_host_tails() {
        let runtime = runtime();
        assert_eq!(runtime.version(), "0.32.3");
        assert!(!runtime.devices().unwrap().is_empty());
        let session = runtime.open_gpu(0).unwrap();
        let prepared = prepared(&session);
        assert_eq!(prepared.compilation_trace_count(), 1);
        let mut prior_outputs = Vec::new();
        for phase in [0.0_f32, 1.0, 2.0] {
            let left = [1.0 + phase, 2.0, 3.0, 4.0, 5.0 + phase, 6.0];
            let right = [7.0, 8.0 + phase, 9.0, 10.0, 11.0, 12.0];
            let a = session.upload_f32([2, 3], &left).unwrap();
            let b = session.upload_f32([3, 2], &right).unwrap();
            assert_eq!(a.residency(), MlxArrayResidency::HostCopied);
            let c = session.execute_matmul(&prepared, &a, &b).unwrap();
            assert_eq!(c.residency(), MlxArrayResidency::GpuEvaluated);
            let mut actual = [-919.0_f32; 7];
            c.read_into_f32(&mut actual).unwrap();
            assert_eq!(c.residency(), MlxArrayResidency::HostMaterialized);
            for row in 0..2 {
                for column in 0..2 {
                    let expected: f32 = (0..3)
                        .map(|k| left[row * 3 + k] * right[k * 2 + column])
                        .sum();
                    assert_eq!(actual[row * 2 + column].to_bits(), expected.to_bits());
                }
            }
            assert_eq!(actual[4..], [-919.0; 3]);
            let mut short = [-313.0_f32; 3];
            assert_eq!(c.read_into_f32(&mut short), Err(MlxError::InvalidExtent));
            assert_eq!(short.map(f32::to_bits), [(-313.0_f32).to_bits(); 3]);
            prior_outputs.push((c, actual[..4].to_vec()));
            for (earlier, expected) in &prior_outputs {
                let mut retained = [0.0; 4];
                earlier.read_into_f32(&mut retained).unwrap();
                assert_eq!(
                    retained.map(f32::to_bits),
                    expected
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                        .as_slice()
                );
            }
        }
    }

    #[test]
    #[ignore = "requires pinned MLX GPU bridge and an external GPU activity check"]
    fn compiled_program_bindings_retain_source_and_retry_before_publication() {
        #[rustfmt::skip]
        use fusion_pcu::dialect::tensor::{
            TensorArithmeticCapability,
            TensorArithmeticRewritePolicy,
            TensorPointwiseGroupingPolicy,
        };
        use std::sync::Arc;
        let runtime = runtime();
        let session = runtime.open_gpu(0).unwrap();
        let mut graph = Graph::default();
        graph.set_numerical_options(PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            ..PcuNumericalOptions::default()
        });
        let a = graph.input([2, 3], PcuScalarType::F32).unwrap();
        let b = graph.input([3, 2], PcuScalarType::F32).unwrap();
        let c = graph.matmul(a, b).unwrap();
        let program = Arc::new(
            graph
                .into_selected_program(
                    &[c],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap(),
        );
        let prepared = session.prepare_program(Arc::clone(&program)).unwrap();
        assert!(std::ptr::eq(prepared.program(), Arc::as_ptr(&program)));
        let left = session.upload_f32([2, 3], &[1.0; 6]).unwrap();
        let right = session.upload_f32([3, 2], &[2.0; 6]).unwrap();
        for inputs in [
            vec![(a, &left)],
            vec![(a, &left), (a, &right)],
            vec![(c, &left), (b, &right)],
        ] {
            assert!(matches!(
                session.execute_program(&prepared, &inputs),
                Err(MlxError::InvalidRequest(_))
            ));
        }
        let retained = prepared.clone();
        drop((prepared, program, runtime));
        let output = session
            .execute_program(&retained, &[(b, &right), (a, &left)])
            .unwrap();
        let mut host = [-13.0; 5];
        output.read_into_f32(&mut host).unwrap();
        assert_eq!(host[..4], [6.0; 4]);
        assert_eq!(host[4].to_bits(), (-13.0_f32).to_bits());
        assert_eq!(retained.matmul().compilation_trace_count(), 1);
    }

    #[test]
    #[ignore = "requires pinned MLX GPU bridge and an external GPU activity check"]
    fn session_shape_guards_retry_and_escaped_array_owners() {
        let runtime = runtime();
        let session = runtime.open_gpu(0).unwrap();
        let foreign = runtime.open_gpu(0).unwrap();
        assert!(session.same_session(&session.clone()));
        assert!(!session.same_session(&foreign));
        session.validate_access_available().unwrap();
        let prepared = prepared(&session);
        let retained_prepared = prepared.clone();
        let a = session.upload_f32([2, 3], &[1.0; 6]).unwrap();
        let b = session.upload_f32([3, 2], &[2.0; 6]).unwrap();
        let foreign_b = foreign.upload_f32([3, 2], &[2.0; 6]).unwrap();
        assert!(matches!(
            session.execute_matmul(&prepared, &a, &foreign_b),
            Err(MlxError::ForeignSession)
        ));
        let wrong = session.upload_f32([2, 2], &[2.0; 4]).unwrap();
        assert!(matches!(
            session.execute_matmul(&prepared, &a, &wrong),
            Err(MlxError::InvalidExtent)
        ));
        assert!(matches!(
            session.upload_f32([0, 3], &[]),
            Err(MlxError::InvalidExtent)
        ));
        drop(prepared);
        let prepared = retained_prepared;
        assert_eq!(prepared.compilation_trace_count(), 1);
        let c = session.execute_matmul(&prepared, &a, &b).unwrap();
        let escaped = c.clone();
        drop((
            c, prepared, a, b, wrong, foreign_b, foreign, session, runtime,
        ));
        let mut result = [0.0; 4];
        escaped.read_into_f32(&mut result).unwrap();
        assert_eq!(result.map(f32::to_bits), [6.0_f32.to_bits(); 4]);
    }

    #[test]
    #[ignore = "requires pinned MLX GPU bridge and an external GPU activity check"]
    fn copied_transport_preserves_encodings_and_native_exceptions_are_explicit() {
        let runtime = runtime();
        let session = runtime.open_gpu(0).unwrap();
        let words = [0_u32, 0x8000_0000, 1, 0x8000_0001, 0x7fc1_2345, 0xff80_0000];
        let data = words.map(f32::from_bits);
        assert!(matches!(
            session.upload_typed([2, 3], &[17_i32; 6]),
            Err(MlxError::UnsupportedScalar(_))
        ));
        let array = session.upload_typed([2, 3], &data).unwrap();
        array.validate_access_available().unwrap();
        let mut actual = [0.0; 6];
        array.read_into_typed(&mut actual).unwrap();
        assert_eq!(actual.map(f32::to_bits), words);
        let mut wrong = [17_i32; 7];
        assert!(matches!(
            array.read_into_typed(&mut wrong),
            Err(MlxError::UnsupportedScalar(_))
        ));
        assert_eq!(wrong, [17; 7]);
        let mut short = [-17.0_f32; 5];
        assert_eq!(
            array.read_into_typed(&mut short),
            Err(MlxError::InvalidExtent)
        );
        assert_eq!(short.map(f32::to_bits), [(-17.0_f32).to_bits(); 5]);
        let mut tailed = [-17.0_f32; 7];
        array.read_into_typed(&mut tailed).unwrap();
        assert_eq!(tailed[6].to_bits(), (-17.0_f32).to_bits());
        let prepared = prepared(&session);
        let b = session.upload_f32([3, 2], &[1.0; 6]).unwrap();
        let result = session.execute_matmul(&prepared, &array, &b).unwrap();
        let mut actual = [-73.0; 5];
        result.read_into_f32(&mut actual).unwrap();
        assert_eq!(actual[4].to_bits(), (-73.0_f32).to_bits());
    }

    #[test]
    #[ignore = "requires pinned MLX GPU bridge and an external GPU activity check"]
    fn discovery_generation_exact_offers_and_prepared_identity() {
        #[rustfmt::skip]
        use fusion_pcu::{
            PcuCostBoundary,
            PcuDeviceActivation,
            PcuDeviceIdentity,
            PcuExecutorId,
            PcuFloatUnderflowPolicy,
            PcuImplementationMechanism,
            PcuImplementationOffers,
            PcuImplementationRequest,
            PcuImplementationRequirements,
            PcuNumericalMode,
            PcuObjectKind,
            PcuRangePolicy,
        };
        let path = std::env::var_os("PCU_MLX_BRIDGE").unwrap();
        let discovery = crate::MlxDiscovery::discover(path).unwrap();
        let reference = discovery.device_reference(0).unwrap();
        let mut stale = reference;
        stale.generation += 1;
        assert!(matches!(
            discovery.open_device(stale),
            Err(MlxError::InvalidRequest(_))
        ));
        stale = reference;
        stale.kind = PcuObjectKind::Context;
        assert!(matches!(
            discovery.open_device(stale),
            Err(MlxError::InvalidRequest(_))
        ));
        let mut graph = Graph::default();
        graph.set_numerical_options(PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            ..PcuNumericalOptions::default()
        });
        let a = graph.input([2, 3], PcuScalarType::F32).unwrap();
        let b = graph.input([3, 2], PcuScalarType::F32).unwrap();
        let c = graph.matmul(a, b).unwrap();
        let node = graph.node(c).unwrap();
        let operation = crate::MlxMatmulRequest {
            graph: &graph,
            node,
        };
        let mut request = PcuImplementationRequest {
            device: PcuDeviceIdentity::from_device_ref(reference).unwrap(),
            executor: PcuExecutorId(0),
            requirements: PcuImplementationRequirements {
                numerical_mode: PcuNumericalMode::Boundary,
                numerical_options: node.numerical_options,
                float_underflow: PcuFloatUnderflowPolicy::IeeeAfterRounding,
                range_policy: PcuRangePolicy::Reject,
            },
            boundary: PcuCostBoundary::Resident,
            operation: &operation,
        };
        assert_eq!(
            discovery.implementation_offers(&request, &mut []).unwrap(),
            1
        );
        let mut offers = [None; 2];
        assert_eq!(
            discovery
                .implementation_offers(&request, &mut offers)
                .unwrap(),
            1
        );
        let offer = offers[0].unwrap();
        assert_eq!(offer.kind, PcuImplementationMechanism::DelegatedRuntime);
        assert_eq!(offer.workspace_bytes, None);
        assert_eq!(offer.cost.completion, None);
        assert_eq!(offers[1], None);
        offer.validate_request(&request).unwrap();
        let session = discovery.open_device(reference).unwrap();
        let prepared = session.prepare_matmul(&graph, node).unwrap();
        assert_eq!(prepared.implementation_id(), Some(offer.implementation));
        request.requirements.range_policy = PcuRangePolicy::Clamp;
        assert_eq!(
            discovery
                .implementation_offers(&request, &mut offers)
                .unwrap(),
            0
        );
        assert_eq!(offers, [None; 2]);
    }
}

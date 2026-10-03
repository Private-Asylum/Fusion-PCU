//! Cold policy admission and real-device numerical/lifetime acceptance for native MSE.
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
};
#[rustfmt::skip]
use super::super::{
    CudaTensorAssessor,
    assess_native_mse_numerical_options,
};

fn options(precision: PcuPrecisionPolicy) -> PcuNumericalOptions {
    PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision,
        reproducibility: PcuReproducibility::Unspecified,
    }
}

#[test]
fn native_mse_independent_policy_matrix_rejects_unproved_combinations_cold() {
    let mut graph = Graph::default();
    let prediction = graph.input([17], PcuScalarType::F32).unwrap();
    let target = graph.input([17], PcuScalarType::F32).unwrap();
    let loss = graph.mean_squared_error(prediction, target).unwrap();
    let environment = crate::CublasEnvironmentSnapshot::capture();
    let mut node = graph.node(loss).unwrap();
    assert_eq!(
        assess_native_mse_numerical_options(node, &environment),
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
        assert!(assess_native_mse_numerical_options(node, &environment).is_ok());
        node.numerical_mode = Some(PcuNumericalMode::Strict);
        assert_eq!(
            assess_native_mse_numerical_options(node, &environment),
            Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::CompoundArithmetic,
                options: options(precision),
            })
        );
        node.numerical_mode = Some(PcuNumericalMode::Boundary);
        node.numerical_options.reproducibility = PcuReproducibility::PortableV1;
        assert!(matches!(
            assess_native_mse_numerical_options(node, &environment),
            Err(TensorUnsupportedReason::NumericalPolicy {
                requirement: PcuNumericalRequirement::Reproducibility,
                ..
            })
        ));
        node.numerical_options = options(precision);
        for policy in [
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            node.float_underflow_policy = Some(policy);
            assert_eq!(
                assess_native_mse_numerical_options(node, &environment),
                Err(TensorUnsupportedReason::UnderflowPolicy(policy))
            );
        }
        node.float_underflow_policy = None;
        node.scalar_type = PcuScalarType::F64;
        assert!(assess_native_mse_numerical_options(node, &environment).is_ok());
        node.scalar_type = PcuScalarType::F32;
    }
}

#[test]
#[ignore = "requires an idle CUDA device and CUDA Toolkit 13.4 cuBLAS component 13.7/13.8"]
#[allow(clippy::too_many_lines)] // One selected session exercises both precision caches, repeated scalar owners and nonfinite permission.
fn native_mse_dense_scalar_output_precision_cache_and_exception_permission() {
    let (_discovery, session) = super::super::tests::cuda_test_session();
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let pool = PcuMemoryPoolId(0x4d53_4543);
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        let mut graph = Graph::default();
        graph.set_numerical_options(options(precision));
        let prediction = graph.input([17], PcuScalarType::F32).unwrap();
        let target = graph.input([17], PcuScalarType::F32).unwrap();
        let loss = graph.mean_squared_error(prediction, target).unwrap();
        assert!(matches!(
            assessor.assess_node(&graph, graph.node(loss).unwrap()),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Library,
                ..
            }
        ));
        let prepared = assessor
            .prepare_owned_program(
                graph
                    .into_selected_program(
                        &[loss],
                        TensorArithmeticRewritePolicy::Disabled,
                        TensorArithmeticCapability::Strict,
                        TensorPointwiseGroupingPolicy::Disabled,
                    )
                    .unwrap(),
            )
            .unwrap();
        for phase in 0..3_u16 {
            let left = core::array::from_fn::<_, 17, _>(|index| {
                f32::from(u16::try_from(index).unwrap() + phase)
            });
            let right = [0.0_f32; 17];
            let sum: u16 = (0..17_u16)
                .map(|index| (index + phase) * (index + phase))
                .sum();
            let expected = f32::from(sum) * super::super::mse_scale(17).unwrap();
            let a =
                PcuDeviceTensor::new([17], session.upload_buffer(pool, &left).unwrap()).unwrap();
            let b =
                PcuDeviceTensor::new([17], session.upload_buffer(pool, &right).unwrap()).unwrap();
            let mut memory = session.memory_provider(pool);
            for _ in 0..2 {
                let output = assessor
                    .execute_owned_program_outputs(
                        &prepared,
                        &[(prediction, &a), (target, &b)],
                        pool,
                        &mut memory,
                    )
                    .unwrap();
                assert!(output[0].1.shape().is_empty());
                let mut actual = [0.0_f32];
                session
                    .download_buffer(pool, output[0].1.buffer(), &mut actual)
                    .unwrap();
                assert_eq!(actual[0].to_bits(), expected.to_bits());
            }
        }
        let a = PcuDeviceTensor::new(
            [17],
            session.upload_buffer(pool, &[f32::INFINITY; 17]).unwrap(),
        )
        .unwrap();
        let b = PcuDeviceTensor::new([17], session.upload_buffer(pool, &[0.0_f32; 17]).unwrap())
            .unwrap();
        let mut memory = session.memory_provider(pool);
        let output = assessor
            .execute_owned_program_outputs(
                &prepared,
                &[(prediction, &a), (target, &b)],
                pool,
                &mut memory,
            )
            .unwrap();
        let mut actual = [0.0_f32];
        session
            .download_buffer(pool, output[0].1.buffer(), &mut actual)
            .unwrap();
        assert!(!actual[0].is_finite());
        assert!(
            assessor
                .native_cublas(PcuScalarType::F32, precision)
                .unwrap()
                .is_usable()
        );
        // Both permitted precision settings retain this exact source-width difference/square
        // witness; optimized permission does not force a reduction in precision for this offer.
        let mut witness = Graph::default();
        witness.set_numerical_options(options(precision));
        let prediction = witness.input([1], PcuScalarType::F32).unwrap();
        let target = witness.input([1], PcuScalarType::F32).unwrap();
        let loss = witness.mean_squared_error(prediction, target).unwrap();
        let prepared = assessor
            .prepare_owned_program(
                witness
                    .into_selected_program(
                        &[loss],
                        TensorArithmeticRewritePolicy::Disabled,
                        TensorArithmeticCapability::Strict,
                        TensorPointwiseGroupingPolicy::Disabled,
                    )
                    .unwrap(),
            )
            .unwrap();
        let a = PcuDeviceTensor::new(
            [1],
            session
                .upload_buffer(pool, &[f32::from_bits(0x3f80_0001)])
                .unwrap(),
        )
        .unwrap();
        let b =
            PcuDeviceTensor::new([1], session.upload_buffer(pool, &[1.0_f32]).unwrap()).unwrap();
        let output = assessor
            .execute_owned_program_outputs(
                &prepared,
                &[(prediction, &a), (target, &b)],
                pool,
                &mut memory,
            )
            .unwrap();
        session
            .download_buffer(pool, output[0].1.buffer(), &mut actual)
            .unwrap();
        assert_eq!(actual[0].to_bits(), ((127_u32 - 46) << 23));
    }
}

#[test]
fn native_f64_mse_uses_double_extents_and_preserved_destination_rounding() {
    let mut graph = Graph::default();
    graph.set_numerical_options(options(PcuPrecisionPolicy::Preserve));
    let p = graph.input([65], PcuScalarType::F64).unwrap();
    let t = graph.input([65], PcuScalarType::F64).unwrap();
    let loss = graph.mean_squared_error(p, t).unwrap();
    let source = super::lower_native_mse_to_cuda_source(&graph, loss).unwrap();
    assert!(source.contains("const double* prediction"));
    assert!(source.contains("__dsub_rn"));
    assert!(source.contains("__dmul_rn"));
    assert!(!source.contains("status"));
    let requirements = super::requirements(65, PcuScalarType::F64);
    assert!(requirements.iter().all(|r| r.min_required_bytes == 520));
    assert!(super::super::mse_scratch_length_fits_scalar(65, PcuScalarType::F64, 520).unwrap());
    assert!(!super::super::mse_scratch_length_fits_scalar(65, PcuScalarType::F64, 519).unwrap());
    assert!(
        super::super::mse_scratch_length_fits_scalar(usize::MAX, PcuScalarType::F64, usize::MAX)
            .is_err()
    );
    assert_eq!(
        super::super::mse_scale_f64(65).unwrap().to_bits(),
        0x3f8f_81f8_1f81_f820
    );
    assert!(super::super::mse_scale_f64(0).is_err());
    graph
        .set_value_numerical_options(loss, PcuNumericalOptions::default())
        .unwrap();
    assert!(super::lower_native_mse_to_cuda_source(&graph, loss).is_err());
}

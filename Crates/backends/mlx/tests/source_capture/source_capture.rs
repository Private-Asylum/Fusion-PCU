//! Authentic annotated-source capture and delegated preparation, distinct from global calls.

#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxMatmulPlan,
    MlxRuntime,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuCompoundArithmeticPolicy,
    PcuExecutionError,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuTensor,
};

#[pcu(crate_path = pcu_facade)]
fn matrix_helper(
    left: &[[f32; 3]; 2],
    right: &[[f32; 2]; 3],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(crate_path = pcu_facade, flag(native_compound), flag(backend_precision))]
fn matrix_source(
    left: &[[f32; 3]; 2],
    right: &[[f32; 2]; 3],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    matrix_helper(left, right)
}

fn captured() -> global::PcuCapturedTensorProgram {
    global::__pcu_capture_tensor_program(
        [
            global::PcuSourceShape::FixedMatrix {
                rows: 2,
                columns: 3,
            },
            global::PcuSourceShape::FixedMatrix {
                rows: 3,
                columns: 2,
            },
        ],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        matrix_source::__pcu_capture_entry,
    )
    .unwrap()
}

#[test]
fn annotated_helper_capture_freezes_the_exact_admitted_matrix_source() {
    let capture = captured();
    let plan = MlxMatmulPlan::assess_program(capture.program()).unwrap();
    assert_eq!(plan.shape(), [2, 2]);
    assert_eq!(plan.inputs().as_slice(), capture.input_values());
    assert_eq!(capture.argument_indices(), [0, 1]);
    assert_eq!(
        plan.numerical_options(),
        PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            precision: PcuPrecisionPolicy::BackendOptimized,
            ..PcuNumericalOptions::default()
        }
    );
}

#[test]
fn annotated_native_permissions_preserve_strict_and_portable_rejection() {
    for (mode, reproducibility) in [
        (PcuNumericalMode::Strict, PcuReproducibility::Unspecified),
        (PcuNumericalMode::Boundary, PcuReproducibility::PortableV1),
    ] {
        let capture = global::__pcu_capture_tensor_program(
            [
                global::PcuSourceShape::FixedMatrix {
                    rows: 2,
                    columns: 3,
                },
                global::PcuSourceShape::FixedMatrix {
                    rows: 3,
                    columns: 2,
                },
            ],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            PcuNumericalOptions {
                reproducibility,
                ..PcuNumericalOptions::default()
            },
            matrix_source::__pcu_capture_entry,
        )
        .unwrap();
        assert!(matches!(
            MlxMatmulPlan::assess_program(capture.program()),
            Err(pcu_facade::dialect::tensor::TensorUnsupportedReason::NumericalPolicy { .. })
        ));
    }
}

#[test]
#[ignore = "requires pinned MLX GPU bridge and an external GPU activity check"]
fn captured_annotated_source_replays_changing_inputs_and_escapes_completed_output() {
    let path = std::env::var_os("PCU_MLX_BRIDGE").expect("set isolated pinned bridge path");
    let runtime = MlxRuntime::load(path).unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let capture = captured();
    let [a, b] = <[_; 2]>::try_from(capture.input_values()).unwrap();
    let prepared = session
        .prepare_program(std::sync::Arc::clone(capture.program()))
        .unwrap();
    assert!(std::ptr::eq(
        prepared.program(),
        std::sync::Arc::as_ptr(capture.program())
    ));
    drop(capture);
    let mut retained = Vec::new();
    for phase in [1.0, 2.0, 3.0] {
        let left = [phase, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0 + phase];
        let left_array = session.upload_f32([2, 3], &left).unwrap();
        let right_array = session.upload_f32([3, 2], &right).unwrap();
        let result = session
            .execute_program(&prepared, &[(a, &left_array), (b, &right_array)])
            .unwrap();
        let expected = std::array::from_fn::<_, 4, _>(|index| {
            let row = index / 2;
            let column = index % 2;
            (0..3)
                .map(|inner| left[row * 3 + inner] * right[inner * 2 + column])
                .sum::<f32>()
        });
        retained.push((result, expected));
    }
    assert_eq!(prepared.matmul().compilation_trace_count(), 1);
    drop((prepared, session, runtime));
    for (output, expected) in retained {
        let mut host = [-41.0; 5];
        output.read_into_f32(&mut host).unwrap();
        assert_eq!(
            host[..4]
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            expected.map(f32::to_bits)
        );
        assert_eq!(host[4].to_bits(), (-41.0_f32).to_bits());
    }
}

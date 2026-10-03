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
#[ignore = "requires pinned safety-patched MLX C library and an external GPU activity check"]
fn captured_annotated_source_replays_changing_inputs_and_escapes_completed_output() {
    let runtime = MlxRuntime::load_default().unwrap();
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

#[pcu(crate_path = pcu_facade, flag(native_compound), flag(backend_precision))]
fn shaped_source<const ROWS: usize, const INNER: usize, const COLUMNS: usize>(
    left: &[[f32; INNER]; ROWS],
    right: &[[f32; COLUMNS]; INNER],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}

fn check_shaped_source<const ROWS: usize, const INNER: usize, const COLUMNS: usize>(
    session: &fusion_pcu_mlx::MlxSession,
) {
    let capture = global::__pcu_capture_tensor_program(
        [
            global::PcuSourceShape::FixedMatrix {
                rows: ROWS,
                columns: INNER,
            },
            global::PcuSourceShape::FixedMatrix {
                rows: INNER,
                columns: COLUMNS,
            },
        ],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        shaped_source::__pcu_capture_entry::<ROWS, INNER, COLUMNS>,
    )
    .unwrap();
    let prepared = session
        .prepare_program(std::sync::Arc::clone(capture.program()))
        .unwrap();
    let [a, b] = prepared.matmul().plan().inputs();
    for phase in [1.0_f32, 2.0] {
        let left: Vec<_> = (0..ROWS * INNER)
            .map(|index| f32::from(u16::try_from(index % 7 + 1).unwrap()) * phase)
            .collect();
        let right: Vec<_> = (0..INNER * COLUMNS)
            .map(|index| f32::from(u16::try_from(index % 5 + 1).unwrap()) + phase)
            .collect();
        let lhs = session.upload_f32([ROWS, INNER], &left).unwrap();
        let rhs = session.upload_f32([INNER, COLUMNS], &right).unwrap();
        let output = session
            .execute_program(&prepared, &[(a, &lhs), (b, &rhs)])
            .unwrap();
        let mut host = vec![-73.0; ROWS * COLUMNS + 1];
        output.read_into_f32(&mut host).unwrap();
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let oracle: f32 = (0..INNER)
                    .map(|inner| left[row * INNER + inner] * right[inner * COLUMNS + column])
                    .sum();
                assert_eq!(host[row * COLUMNS + column].to_bits(), oracle.to_bits());
            }
        }
        assert_eq!(host[ROWS * COLUMNS].to_bits(), (-73.0_f32).to_bits());
    }
    assert_eq!(prepared.matmul().compilation_trace_count(), 1);
}

#[test]
#[ignore = "requires exact production C library and an external GPU activity check"]
fn captured_singleton_and_rectangular_shape_borders() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    check_shaped_source::<1, 1, 1>(&session);
    check_shaped_source::<1, 3, 1>(&session);
    check_shaped_source::<1, 3, 4>(&session);
    check_shaped_source::<4, 3, 1>(&session);
    check_shaped_source::<3, 1, 5>(&session);
    check_shaped_source::<7, 13, 5>(&session);
    check_shaped_source::<3, 7, 9>(&session);
}

#[test]
#[ignore = "requires pinned MLX C GPU and external activity guard"]
fn opaque_prepared_source_stages_ram_and_rejects_foreign_before_publication() {
    #[rustfmt::skip]
    use fusion_pcu_mlx::{
        MlxError,
        MlxProgramInput,
    };
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let other = runtime.open_gpu(0).unwrap();
    let capture = captured();
    let prepared = session
        .prepare_program(std::sync::Arc::clone(capture.program()))
        .unwrap();
    let [left_id, right_id] = prepared.matmul().plan().inputs();
    let left = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let resident = session.upload_f32([3, 2], &right).unwrap();
    let foreign = other.upload_f32([3, 2], &right).unwrap();
    let mut host = [-73.0_f32; 5];
    assert!(matches!(
        prepared.execute_mixed(&[
            (left_id, MlxProgramInput::Host(&left)),
            (right_id, MlxProgramInput::Resident(&foreign))
        ]),
        Err(MlxError::ForeignSession)
    ));
    assert!(matches!(
        prepared.execute_host_into(&[(left_id, &left[..5]), (right_id, &right)], &mut host),
        Err(MlxError::InvalidExtent)
    ));
    assert_eq!(host.map(f32::to_bits), [-73.0_f32; 5].map(f32::to_bits));
    assert!(matches!(
        prepared.execute_host_into(&[(left_id, &left[..]), (left_id, &right[..])], &mut host),
        Err(MlxError::InvalidRequest(_))
    ));
    assert!(matches!(
        prepared.execute_host_into(
            &[(left_id, &[1_u32; 6]), (right_id, &[1_u32; 6])],
            &mut [0_u32; 4]
        ),
        Err(MlxError::UnsupportedScalar(_))
    ));
    prepared
        .execute_host_into(&[(right_id, &right[..]), (left_id, &left[..])], &mut host)
        .unwrap();
    assert_eq!(
        host.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0, -73.0].map(f32::to_bits)
    );
    let output = prepared
        .execute_mixed(&[
            (right_id, MlxProgramInput::Resident(&resident)),
            (left_id, MlxProgramInput::Host(&left)),
        ])
        .unwrap();
    assert_eq!(prepared.matmul().compilation_trace_count(), 1);
    drop((
        prepared, capture, session, other, runtime, resident, foreign,
    ));
    output.read_into_f32(&mut host).unwrap();
    assert_eq!(
        host.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0, -73.0].map(f32::to_bits)
    );
}

//! Annotated source numerical contracts, cache changes, and escaped device outputs.
#![cfg(feature = "tensor")]

#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[cfg(any(feature = "rocm", feature = "cuda"))]
#[rustfmt::skip]
use fusion_pcu::{
    dialect::tensor::{TensorArithmeticStep, TensorError},
    global,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuRangePolicy,
};

#[pcu]
fn inherited<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    lhs: &[[T; K]; R],
    rhs: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(lhs, rhs)
}

#[pcu(flag(strict))]
fn strict<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    lhs: &[[T; K]; R],
    rhs: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    inherited::<T, R, K, C>(lhs, rhs)
}

#[pcu(flag(non_strict))]
fn boundary<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    lhs: &[[T; K]; R],
    rhs: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    inherited::<T, R, K, C>(lhs, rhs)
}

#[pcu(flag(non_strict))]
fn discarded_boundary<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    lhs: &[[T; K]; R],
    rhs: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _discarded = pcu::matmul(lhs, rhs)?;
    pcu::identity(lhs)
}

#[pcu(flag(strict))]
fn locally_disabled<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    lhs: &[[T; K]; R],
    rhs: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    boundary::<T, R, K, C>(lhs, rhs)
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
fn configure(
    backend: global::PcuBackendChoice,
    numerical_mode: PcuNumericalMode,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) {
    global::configure(global::PcuExecutionPolicy {
        backend,
        numerical_mode,
        float_underflow: underflow,
        range_policy: range,
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
fn assert_boundary_rejection(error: &PcuExecutionError) {
    let unsupported = |rejection: &PcuExecutionError| match rejection {
        #[cfg(feature = "rocm")]
        PcuExecutionError::TensorExecution(
            fusion_pcu_rocm::RocmTensorExecutionError::Unsupported { .. },
        ) => true,
        #[cfg(feature = "cuda")]
        PcuExecutionError::CudaTensorExecution(
            fusion_pcu_cuda::CudaTensorExecutionError::Unsupported { .. },
        ) => true,
        _ => false,
    };
    let rejected = match error {
        #[cfg(feature = "rocm")]
        PcuExecutionError::NoCompatibleDevice(rejections) => {
            rejections.iter().any(|(_, error)| unsupported(error))
        }
        PcuExecutionError::NoCompatibleResidentDevice(rejections) => {
            rejections.iter().any(|(_, error)| unsupported(error))
        }
        _ => false,
    };
    assert!(
        rejected,
        "expected unsupported compound contract with provider metadata: {error:?}"
    );
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
fn assert_fault(
    error: &PcuExecutionError,
    expected_element: usize,
    expected_reduction: usize,
    expected_step: TensorArithmeticStep,
    expected_kind: PcuExecutionFaultKind,
) {
    let graph_error = match error {
        #[cfg(feature = "rocm")]
        PcuExecutionError::TensorExecution(fusion_pcu_rocm::RocmTensorExecutionError::Graph(
            error,
        )) => error,
        #[cfg(feature = "cuda")]
        PcuExecutionError::CudaTensorExecution(
            fusion_pcu_cuda::CudaTensorExecutionError::Graph(error),
        ) => error,
        _ => panic!("expected compound fault: {error:?}"),
    };
    let TensorError::CompoundArithmeticFault {
        element_index,
        reduction_index,
        step,
        kind,
        ..
    } = graph_error
    else {
        panic!("expected compound location metadata: {graph_error:?}");
    };
    assert_eq!(*element_index, expected_element);
    assert_eq!(*reduction_index, expected_reduction);
    assert_eq!(*step, expected_step);
    assert_eq!(*kind, expected_kind);
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[allow(clippy::too_many_lines)] // One serial hardware scenario controls the process policy.
fn source_contracts(backend: global::PcuBackendChoice) {
    configure(
        backend,
        PcuNumericalMode::Boundary,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    global::clear_thread_cache().unwrap();
    let lhs = [[1.0_f32, 2.0], [3.0, 4.0]];
    let rhs = [[5.0_f32, 6.0], [7.0, 8.0]];
    assert_boundary_rejection(&inherited(&lhs, &rhs).unwrap_err());
    assert_boundary_rejection(&discarded_boundary(&lhs, &rhs).unwrap_err());
    let escaped = strict(&lhs, &rhs).unwrap();
    let mut output = [0.0_f32; 4];
    escaped.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [19.0_f32, 22.0, 43.0, 50.0].map(f32::to_bits)
    );
    let resident = strict::<f32, 2, 2, 2>(&escaped, &rhs).unwrap();
    resident.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [249.0_f32, 290.0, 565.0, 658.0].map(f32::to_bits)
    );

    configure(
        backend,
        PcuNumericalMode::Strict,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    inherited(&lhs, &rhs)
        .unwrap()
        .read_into(&mut output)
        .unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [19.0_f32, 22.0, 43.0, 50.0].map(f32::to_bits)
    );
    assert_boundary_rejection(&boundary(&lhs, &rhs).unwrap_err());
    assert_boundary_rejection(&discarded_boundary(&lhs, &rhs).unwrap_err());
    assert_boundary_rejection(&locally_disabled(&lhs, &rhs).unwrap_err());
    configure(
        backend,
        PcuNumericalMode::Boundary,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    assert_boundary_rejection(&inherited(&lhs, &rhs).unwrap_err());

    // Overflow is reported before a later term could cancel it. Row-major output selection
    // precedes reduction-step selection even when multiple lanes fail.
    let cancellation = [[1.0_f32, 1.0, 1.0], [f32::MAX, f32::MAX, -f32::MAX]];
    let ones = [[1.0_f32], [1.0], [1.0]];
    assert_fault(
        &strict(&cancellation, &ones).unwrap_err(),
        1,
        1,
        TensorArithmeticStep::Add,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    strict(&[[1.0_f32; 3]; 2], &ones)
        .unwrap()
        .read_into(&mut [0.0_f32; 2])
        .unwrap();
    let simultaneous = [[1.0_f32, f32::MAX], [f32::MAX, f32::MAX]];
    assert_fault(
        &strict(&simultaneous, &[[2.0_f32], [2.0]]).unwrap_err(),
        0,
        1,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );

    let tiny = [[f32::from_bits(1)]];
    assert_fault(
        &strict(&tiny, &[[0.5_f32]]).unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    let mut single = [0.0_f32];
    strict(&tiny, &[[1.0_f32]])
        .unwrap()
        .read_into(&mut single)
        .unwrap();
    assert_eq!(single[0].to_bits(), 1);
    configure(
        backend,
        PcuNumericalMode::Strict,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuRangePolicy::Reject,
    );
    inherited(&tiny, &[[0.5_f32]])
        .unwrap()
        .read_into(&mut single)
        .unwrap();
    assert_eq!(single[0].to_bits(), 0);
    configure(
        backend,
        PcuNumericalMode::Strict,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuRangePolicy::Reject,
    );
    assert_fault(
        &inherited(&tiny, &[[1.0_f32]]).unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );

    configure(
        backend,
        PcuNumericalMode::Boundary,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    let double = [[16_777_217.0_f64, 1.0e-10], [-3.0, 0.5]];
    let identity = [[1.0_f64, 0.0], [0.0, 1.0]];
    let double_result = strict(&double, &identity).unwrap();
    let mut observed = [0.0_f64; 4];
    double_result.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f64::to_bits),
        [double[0][0], double[0][1], double[1][0], double[1][1]].map(f64::to_bits)
    );
    assert_fault(
        &strict(&[[f64::from_bits(1)]], &[[0.5_f64]]).unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    assert_fault(
        &strict(&[[f64::MAX, f64::MAX, -f64::MAX]], &[[1.0_f64]; 3]).unwrap_err(),
        0,
        1,
        TensorArithmeticStep::Add,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    strict(&double, &identity)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();

    #[cfg(feature = "cuda")]
    {
        let other = if backend == global::PcuBackendChoice::Cuda {
            global::PcuBackendChoice::Rocm
        } else {
            global::PcuBackendChoice::Cuda
        };
        configure(
            other,
            PcuNumericalMode::Strict,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Reject,
        );
        assert!(matches!(
            strict::<f32, 2, 2, 2>(&escaped, &rhs),
            Err(PcuExecutionError::ResidentPolicyConflict)
        ));
        configure(
            backend,
            PcuNumericalMode::Strict,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Reject,
        );
        strict(&lhs, &rhs).unwrap().read_into(&mut output).unwrap();
        assert_eq!(
            output.map(f32::to_bits),
            [19.0_f32, 22.0, 43.0, 50.0].map(f32::to_bits)
        );
    }

    configure(
        backend,
        PcuNumericalMode::Strict,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Clamp,
    );
    assert!(matches!(
        strict(&lhs, &rhs),
        Err(PcuExecutionError::UnsupportedRangePolicy)
    ));
    configure(
        backend,
        PcuNumericalMode::Boundary,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    global::clear_thread_cache().unwrap();
    escaped.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [19.0_f32, 22.0, 43.0, 50.0].map(f32::to_bits)
    );
    double_result.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f64::to_bits),
        [double[0][0], double[0][1], double[1][0], double[1][1]].map(f64::to_bits)
    );
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware; run serially"]
fn rocm_annotated_strict_matmul_contracts() {
    source_contracts(global::PcuBackendChoice::Rocm);
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires CUDA hardware; run serially"]
fn cuda_annotated_strict_matmul_contracts() {
    source_contracts(global::PcuBackendChoice::Cuda);
}

#[test]
#[cfg(not(any(feature = "rocm", feature = "cuda")))]
fn strict_source_has_no_host_fallback() {
    assert!(matches!(
        strict(&[[1.0_f32]], &[[2.0_f32]]),
        Err(PcuExecutionError::TensorExecutionUnavailable)
    ));
}

//! Actual-device value/fault parity; run only on an idle authorized CUDA device.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuScalar,
};
#[rustfmt::skip]
use crate::{
    CudaOwnedDispatchBackend,
    CudaOwnedPreparedTensorGraph,
    CudaTensorAssessor,
};

struct Case {
    prepared: CudaOwnedPreparedTensorGraph,
    left: ValueId,
    right: ValueId,
    product: ValueId,
    shapes: [[usize; 2]; 2],
    output_count: usize,
}

impl Case {
    fn new(
        assessor: &CudaTensorAssessor<'_>,
        scalar: PcuScalarType,
        shapes: [[usize; 2]; 2],
        transpose: [bool; 2],
        policy: PcuFloatUnderflowPolicy,
        dependent_relu: bool,
    ) -> Self {
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let left = graph.input(shapes[0], scalar).unwrap();
        let right = graph.input(shapes[1], scalar).unwrap();
        let product = graph
            .matmul_transposed(left, right, transpose[0], transpose[1])
            .unwrap();
        graph
            .set_value_float_underflow_policy(product, policy)
            .unwrap();
        let output_count = graph.shape(product).unwrap().iter().product();
        let output = if dependent_relu {
            graph.relu(product).unwrap()
        } else {
            product
        };
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let prepared = assessor.prepare_owned_program(program).unwrap();
        assert!(!prepared.data.requires_blas);
        Self {
            prepared,
            left,
            right,
            product,
            shapes,
            output_count,
        }
    }

    fn run<T: PcuScalar>(
        &self,
        assessor: &CudaTensorAssessor<'_>,
        session: &CudaOwnedDispatchBackend,
        left: &[T],
        right: &[T],
    ) -> Result<Vec<T>, CudaTensorExecutionError> {
        let pool = PcuMemoryPoolId(0x5354_4355);
        let left_tensor =
            PcuDeviceTensor::new(self.shapes[0], session.upload_buffer(pool, left).unwrap())
                .unwrap();
        let right_tensor =
            PcuDeviceTensor::new(self.shapes[1], session.upload_buffer(pool, right).unwrap())
                .unwrap();
        let mut memory = session.memory_provider(pool);
        let outputs = assessor.execute_owned_program_outputs(
            &self.prepared,
            &[(self.left, &left_tensor), (self.right, &right_tensor)],
            pool,
            &mut memory,
        )?;
        assert_eq!(outputs.len(), 1);
        let mut actual = vec![left[0]; self.output_count];
        session
            .download_buffer(pool, outputs[0].1.buffer(), &mut actual)
            .unwrap();
        Ok(actual)
    }

    fn fault(
        &self,
        error: &CudaTensorExecutionError,
        element: usize,
        k: usize,
        step: TensorArithmeticStep,
        kind: PcuExecutionFaultKind,
    ) {
        assert!(
            matches!(error, CudaTensorExecutionError::Graph(TensorError::CompoundArithmeticFault {
            value, element_index, reduction_index, step: actual_step, kind: actual_kind,
        }) if *value == self.product && *element_index == element && *reduction_index == k
            && *actual_step == step && *actual_kind == kind)
        );
    }
}

#[test]
#[ignore = "requires a working idle CUDA device"]
fn strict_matmul_f32_ordered_range_policy_transpose_fault_retry_and_dependency() {
    let (_discovery, session) = super::super::super::tests::cuda_test_session();
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    public_source_compiles_with_nvrtc(&session, PcuScalarType::F32);
    f32_transpose(&assessor, &session);
    f32_arithmetic(&assessor, &session);
    f32_policies(&assessor, &session);
    f32_order_and_partial(&assessor, &session);
}

#[test]
#[ignore = "requires a working idle CUDA device"]
fn strict_matmul_f64_ordered_range_policy_transpose_fault_retry_and_dependency() {
    let (_discovery, session) = super::super::super::tests::cuda_test_session();
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    public_source_compiles_with_nvrtc(&session, PcuScalarType::F64);
    f64_transpose(&assessor, &session);
    f64_arithmetic(&assessor, &session);
    f64_policies(&assessor, &session);
    f64_order_and_partial(&assessor, &session);
}

fn public_source_compiles_with_nvrtc(session: &CudaOwnedDispatchBackend, scalar: PcuScalarType) {
    let (graph, output) = graph(scalar);
    let source = lower_strict_matmul_to_cuda_source(&graph, output).unwrap();
    let image = crate::compile_cuda_source_for_device(session.tensor_runtime(), &source).unwrap();
    assert!(!image.is_empty());
}

fn f32_transpose(assessor: &CudaTensorAssessor<'_>, session: &CudaOwnedDispatchBackend) {
    let ieee = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    for transpose in [[false, false], [true, false], [false, true], [true, true]] {
        let shapes = [
            if transpose[0] { [3, 2] } else { [2, 3] },
            if transpose[1] { [2, 3] } else { [3, 2] },
        ];
        let left = if transpose[0] {
            [1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]
        } else {
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        };
        let right = if transpose[1] {
            [7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0]
        } else {
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        };
        let case = Case::new(assessor, PcuScalarType::F32, shapes, transpose, ieee, false);
        assert_eq!(
            case.run::<f32>(assessor, session, &left, &right).unwrap(),
            [58.0, 64.0, 139.0, 154.0]
        );
    }
}

fn f32_arithmetic(assessor: &CudaTensorAssessor<'_>, session: &CudaOwnedDispatchBackend) {
    let ieee = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let dot = Case::new(
        assessor,
        PcuScalarType::F32,
        [[1, 2], [2, 1]],
        [false; 2],
        ieee,
        true,
    );
    let tiny = f32::from_bits(1);
    dot.fault(
        &dot.run::<f32>(assessor, session, &[tiny, 1.0], &[0.5, 1.0])
            .unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    assert_eq!(
        dot.run::<f32>(assessor, session, &[2.0, 3.0], &[4.0, 5.0])
            .unwrap(),
        [23.0]
    );
    dot.fault(
        &dot.run::<f32>(assessor, session, &[f32::MAX, f32::MAX], &[1.0, 1.0])
            .unwrap_err(),
        0,
        1,
        TensorArithmeticStep::Add,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    dot.fault(
        &dot.run::<f32>(assessor, session, &[f32::MAX, f32::MAX], &[2.0, -2.0])
            .unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    dot.fault(
        &dot.run::<f32>(assessor, session, &[1.0, f32::NAN], &[1.0, 1.0])
            .unwrap_err(),
        0,
        1,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    let no_fma = Case::new(
        assessor,
        PcuScalarType::F32,
        [[1, 2], [2, 1]],
        [false; 2],
        ieee,
        false,
    );
    assert_eq!(
        no_fma
            .run::<f32>(
                assessor,
                session,
                &[-1.0, f32::from_bits(0x3f80_0001)],
                &[1.0, f32::from_bits(0x3f7f_fffe)]
            )
            .unwrap()[0]
            .to_bits(),
        0
    );
}

fn f32_policies(assessor: &CudaTensorAssessor<'_>, session: &CudaOwnedDispatchBackend) {
    let ieee = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let tiny = f32::from_bits(1);
    let exact = Case::new(
        assessor,
        PcuScalarType::F32,
        [[1, 1], [1, 1]],
        [false; 2],
        ieee,
        false,
    );
    assert_eq!(
        exact
            .run::<f32>(assessor, session, &[tiny], &[1.0])
            .unwrap()[0]
            .to_bits(),
        1
    );
    assert_eq!(
        exact
            .run::<f32>(assessor, session, &[-0.0_f32], &[1.0])
            .unwrap()[0]
            .to_bits(),
        0
    );
    let gradual = Case::new(
        assessor,
        PcuScalarType::F32,
        [[1, 2], [2, 1]],
        [false; 2],
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        false,
    );
    assert_eq!(
        gradual
            .run::<f32>(assessor, session, &[tiny, 1.0], &[0.5, 1.0])
            .unwrap(),
        [1.0]
    );
    let reject = Case::new(
        assessor,
        PcuScalarType::F32,
        [[1, 1], [1, 1]],
        [false; 2],
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        false,
    );
    reject.fault(
        &reject
            .run::<f32>(assessor, session, &[tiny], &[1.0])
            .unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
}

fn f32_order_and_partial(assessor: &CudaTensorAssessor<'_>, session: &CudaOwnedDispatchBackend) {
    let ieee = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let partial = Case::new(
        assessor,
        PcuScalarType::F32,
        [[65, 2], [2, 1]],
        [false; 2],
        ieee,
        false,
    );
    for phase in [2.0_f32, 7.0] {
        let left: Vec<_> = std::iter::repeat_n([phase, 3.0], 65).flatten().collect();
        assert_eq!(
            partial
                .run::<f32>(assessor, session, &left, &[1.0, 2.0])
                .unwrap(),
            vec![phase + 6.0; 65]
        );
    }
    let ordered = Case::new(
        assessor,
        PcuScalarType::F32,
        [[1, 3], [3, 1]],
        [false; 2],
        ieee,
        false,
    );
    assert_eq!(
        ordered
            .run::<f32>(
                assessor,
                session,
                &[16_777_216.0_f32, 1.0, -16_777_216.0],
                &[1.0; 3]
            )
            .unwrap()[0]
            .to_bits(),
        0
    );
    let earliest = Case::new(
        assessor,
        PcuScalarType::F32,
        [[2, 2], [2, 1]],
        [false; 2],
        ieee,
        false,
    );
    earliest.fault(
        &earliest
            .run::<f32>(
                assessor,
                session,
                &[1.0, f32::NAN, f32::INFINITY, 1.0],
                &[1.0; 2],
            )
            .unwrap_err(),
        0,
        1,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    assert_eq!(
        earliest
            .run::<f32>(assessor, session, &[1.0, 2.0, 3.0, 4.0], &[1.0; 2])
            .unwrap(),
        [3.0, 7.0]
    );
}

fn f64_transpose(assessor: &CudaTensorAssessor<'_>, session: &CudaOwnedDispatchBackend) {
    let ieee = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let transposed = Case::new(
        assessor,
        PcuScalarType::F64,
        [[3, 2], [2, 3]],
        [true; 2],
        ieee,
        false,
    );
    assert_eq!(
        transposed
            .run::<f64>(
                assessor,
                session,
                &[1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0],
                &[7.0, 9.0, 11.0, 8.0, 10.0, 12.0]
            )
            .unwrap(),
        [58.0, 64.0, 139.0, 154.0]
    );
}

fn f64_arithmetic(assessor: &CudaTensorAssessor<'_>, session: &CudaOwnedDispatchBackend) {
    let ieee = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let dot = Case::new(
        assessor,
        PcuScalarType::F64,
        [[1, 2], [2, 1]],
        [false; 2],
        ieee,
        true,
    );
    let tiny = f64::from_bits(1);
    dot.fault(
        &dot.run::<f64>(assessor, session, &[tiny, 1.0], &[0.5, 1.0])
            .unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    assert_eq!(
        dot.run::<f64>(assessor, session, &[2.0, 3.0], &[4.0, 5.0])
            .unwrap(),
        [23.0]
    );
    dot.fault(
        &dot.run::<f64>(assessor, session, &[f64::MAX, f64::MAX], &[1.0, 1.0])
            .unwrap_err(),
        0,
        1,
        TensorArithmeticStep::Add,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    dot.fault(
        &dot.run::<f64>(assessor, session, &[f64::MAX, f64::MAX], &[2.0, -2.0])
            .unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    dot.fault(
        &dot.run::<f64>(assessor, session, &[1.0, f64::INFINITY], &[1.0, 1.0])
            .unwrap_err(),
        0,
        1,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    let no_fma = Case::new(
        assessor,
        PcuScalarType::F64,
        [[1, 2], [2, 1]],
        [false; 2],
        ieee,
        false,
    );
    assert_eq!(
        no_fma
            .run::<f64>(
                assessor,
                session,
                &[-1.0, f64::from_bits(0x3ff0_0000_0000_0001)],
                &[1.0, f64::from_bits(0x3fef_ffff_ffff_fffe)]
            )
            .unwrap()[0]
            .to_bits(),
        0
    );
}

fn f64_policies(assessor: &CudaTensorAssessor<'_>, session: &CudaOwnedDispatchBackend) {
    let ieee = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let tiny = f64::from_bits(1);
    let exact = Case::new(
        assessor,
        PcuScalarType::F64,
        [[1, 1], [1, 1]],
        [false; 2],
        ieee,
        false,
    );
    assert_eq!(
        exact
            .run::<f64>(assessor, session, &[tiny], &[1.0])
            .unwrap()[0]
            .to_bits(),
        1
    );
    assert_eq!(
        exact
            .run::<f64>(assessor, session, &[-0.0_f64], &[1.0])
            .unwrap()[0]
            .to_bits(),
        0
    );
    let gradual = Case::new(
        assessor,
        PcuScalarType::F64,
        [[1, 2], [2, 1]],
        [false; 2],
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        false,
    );
    assert_eq!(
        gradual
            .run::<f64>(assessor, session, &[tiny, 1.0], &[0.5, 1.0])
            .unwrap(),
        [1.0]
    );
    let reject = Case::new(
        assessor,
        PcuScalarType::F64,
        [[1, 1], [1, 1]],
        [false; 2],
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        false,
    );
    reject.fault(
        &reject
            .run::<f64>(assessor, session, &[tiny], &[1.0])
            .unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
}

fn f64_order_and_partial(assessor: &CudaTensorAssessor<'_>, session: &CudaOwnedDispatchBackend) {
    let ieee = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let ordered = Case::new(
        assessor,
        PcuScalarType::F64,
        [[1, 3], [3, 1]],
        [false; 2],
        ieee,
        false,
    );
    assert_eq!(
        ordered
            .run::<f64>(
                assessor,
                session,
                &[9_007_199_254_740_992.0_f64, 1.0, -9_007_199_254_740_992.0],
                &[1.0; 3]
            )
            .unwrap()[0]
            .to_bits(),
        0
    );
    let partial = Case::new(
        assessor,
        PcuScalarType::F64,
        [[65, 2], [2, 1]],
        [false; 2],
        ieee,
        false,
    );
    for phase in [2.0_f64, 7.0] {
        let left: Vec<_> = std::iter::repeat_n([phase, 3.0], 65).flatten().collect();
        assert_eq!(
            partial
                .run::<f64>(assessor, session, &left, &[1.0, 2.0])
                .unwrap(),
            vec![phase + 6.0; 65]
        );
    }
}

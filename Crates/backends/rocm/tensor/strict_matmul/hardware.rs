//! Device parity for the checked reference law, ordered faults, and warm input changes.
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceTensor,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuMemoryPoolId,
    PcuNumericalMode,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    Tensor,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorArithmeticStep,
    TensorElement,
    TensorError,
    TensorPointwiseGroupingPolicy,
    ValueId,
};
#[rustfmt::skip]
use crate::{
    RocmOwnedDispatchBackend,
    RocmOwnedPreparedTensorGraph,
    RocmTensorAssessor,
    RocmTensorExecutionError,
};

trait Scalar: TensorElement + Default + core::fmt::Debug {
    fn small(value: i16) -> Self;
    fn bits(self) -> u64;
    fn max() -> Self;
    fn tiny() -> Self;
    fn half() -> Self;
    fn no_fma_witness() -> ([Self; 2], [Self; 2]);
}

impl Scalar for f32 {
    fn small(value: i16) -> Self {
        Self::from(value)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn max() -> Self {
        Self::MAX
    }
    fn tiny() -> Self {
        Self::from_bits(1)
    }
    fn half() -> Self {
        0.5
    }
    fn no_fma_witness() -> ([Self; 2], [Self; 2]) {
        (
            [-1.0, Self::from_bits(0x3f80_0001)],
            [1.0, Self::from_bits(0x3f7f_fffe)],
        )
    }
}

impl Scalar for f64 {
    fn small(value: i16) -> Self {
        Self::from(value)
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn max() -> Self {
        Self::MAX
    }
    fn tiny() -> Self {
        Self::from_bits(1)
    }
    fn half() -> Self {
        0.5
    }
    fn no_fma_witness() -> ([Self; 2], [Self; 2]) {
        (
            [-1.0, Self::from_bits(0x3ff0_0000_0000_0001)],
            [1.0, Self::from_bits(0x3fef_ffff_ffff_fffe)],
        )
    }
}

struct Case {
    prepared: RocmOwnedPreparedTensorGraph,
    inputs: [ValueId; 2],
    product: ValueId,
    output: ValueId,
    shapes: [[usize; 2]; 2],
}

impl Case {
    fn new<T: Scalar>(
        assessor: &RocmTensorAssessor<'_>,
        shapes: [[usize; 2]; 2],
        transpose: [bool; 2],
        dependent_relu: bool,
    ) -> Self {
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let inputs = shapes.map(|shape| graph.input(shape, T::TYPE).unwrap());
        let product = graph
            .matmul_transposed(inputs[0], inputs[1], transpose[0], transpose[1])
            .unwrap();
        graph
            .set_value_float_underflow_policy(product, PcuFloatUnderflowPolicy::IeeeAfterRounding)
            .unwrap();
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
        Self {
            prepared,
            inputs,
            product,
            output,
            shapes,
        }
    }

    fn parity<T: Scalar>(
        &self,
        assessor: &RocmTensorAssessor<'_>,
        session: &RocmOwnedDispatchBackend,
        left: &[T],
        right: &[T],
    ) -> Result<Vec<T>, TensorError> {
        let reference = self.prepared.graph().evaluate_checked(&[
            (
                self.inputs[0],
                T::into_value(Tensor::new(self.shapes[0], left.to_vec()).unwrap()),
            ),
            (
                self.inputs[1],
                T::into_value(Tensor::new(self.shapes[1], right.to_vec()).unwrap()),
            ),
        ]);
        let pool = PcuMemoryPoolId(0x5354_524f);
        let left_tensor =
            PcuDeviceTensor::new(self.shapes[0], session.upload_buffer(pool, left).unwrap())
                .unwrap();
        let right_tensor =
            PcuDeviceTensor::new(self.shapes[1], session.upload_buffer(pool, right).unwrap())
                .unwrap();
        let mut memory = session.memory_provider(pool);
        let actual = assessor.execute_owned_program_outputs(
            &self.prepared,
            &[
                (self.inputs[0], &left_tensor),
                (self.inputs[1], &right_tensor),
            ],
            pool,
            &mut memory,
        );
        match (reference, actual) {
            (Ok(reference), Ok(outputs)) => {
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0].0, self.output);
                let expected = reference.value_typed::<T>(self.output).unwrap().data();
                let mut observed = vec![T::default(); expected.len()];
                session
                    .download_buffer(pool, outputs[0].1.buffer(), &mut observed)
                    .unwrap();
                for (index, (expected, observed)) in expected.iter().zip(&observed).enumerate() {
                    assert_eq!(expected.bits(), observed.bits(), "cell {index}");
                }
                Ok(observed)
            }
            (Err(expected), Err(RocmTensorExecutionError::Graph(actual))) => {
                assert_eq!(expected, actual);
                Err(actual)
            }
            (reference, actual) => {
                let actual = actual.map(|outputs| outputs.len());
                panic!("reference/device mismatch: {reference:?}; {actual:?}")
            }
        }
    }

    fn fault(
        &self,
        error: &TensorError,
        element: usize,
        reduction: usize,
        step: TensorArithmeticStep,
        kind: PcuExecutionFaultKind,
    ) {
        assert_eq!(
            error,
            &TensorError::CompoundArithmeticFault {
                value: self.product,
                element_index: element,
                reduction_index: reduction,
                step,
                kind,
            }
        );
    }
}

fn transposes<T: Scalar>(assessor: &RocmTensorAssessor<'_>, session: &RocmOwnedDispatchBackend) {
    for transpose in [[false, false], [false, true], [true, false], [true, true]] {
        let shapes = [
            if transpose[0] { [3, 2] } else { [2, 3] },
            if transpose[1] { [2, 3] } else { [3, 2] },
        ];
        let left = if transpose[0] {
            [1, 4, 2, 5, 3, 6]
        } else {
            [1, 2, 3, 4, 5, 6]
        }
        .map(T::small);
        let right = if transpose[1] {
            [7, 9, 11, 8, 10, 12]
        } else {
            [7, 8, 9, 10, 11, 12]
        }
        .map(T::small);
        let case = Case::new::<T>(assessor, shapes, transpose, false);
        case.parity(assessor, session, &left, &right).unwrap();
        let changed = left.map(|_| T::small(2));
        case.parity(assessor, session, &changed, &right).unwrap();
    }
}

fn ordered_faults<T: Scalar>(
    assessor: &RocmTensorAssessor<'_>,
    session: &RocmOwnedDispatchBackend,
) {
    // The dependent ReLU output is never published on a product fault. Retrying the same
    // prepared graph succeeds, proving prior fault status does not poison the next call.
    let case = Case::new::<T>(assessor, [[2, 3], [3, 1]], [false; 2], true);
    let left = [
        T::small(1),
        T::small(1),
        T::small(1),
        T::max(),
        T::max(),
        T::small(0),
    ];
    let right = [T::small(1); 3];
    case.fault(
        &case.parity(assessor, session, &left, &right).unwrap_err(),
        1,
        1,
        TensorArithmeticStep::Add,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    case.parity(assessor, session, &[T::small(2); 6], &right)
        .unwrap();

    let priority = [
        T::small(1),
        T::max(),
        T::max(),
        T::max(),
        T::max(),
        T::max(),
    ];
    case.fault(
        &case
            .parity(assessor, session, &priority, &[T::small(2); 3])
            .unwrap_err(),
        0,
        1,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    case.parity(assessor, session, &[T::small(-2); 6], &right)
        .unwrap();

    let dot = Case::new::<T>(assessor, [[1, 2], [2, 1]], [false; 2], true);
    dot.fault(
        &dot.parity(
            assessor,
            session,
            &[T::tiny(), T::small(1)],
            &[T::half(), T::small(1)],
        )
        .unwrap_err(),
        0,
        0,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    dot.parity(
        assessor,
        session,
        &[T::tiny(), T::small(0)],
        &[T::small(1); 2],
    )
    .unwrap();
    let no_fma = Case::new::<T>(assessor, [[1, 2], [2, 1]], [false; 2], false);
    let (left, right) = T::no_fma_witness();
    assert_eq!(
        no_fma.parity(assessor, session, &left, &right).unwrap()[0].bits(),
        0
    );
}

fn partial_block<T: Scalar>(assessor: &RocmTensorAssessor<'_>, session: &RocmOwnedDispatchBackend) {
    let case = Case::new::<T>(assessor, [[65, 3], [3, 1]], [false; 2], false);
    let borrowed = assessor
        .prepare_graph(case.prepared.graph(), case.output)
        .unwrap();
    assert!(!borrowed.requires_blas);
    let initial = assessor.prewarm_prepared_graph(&borrowed).unwrap();
    assert_eq!(initial.requested_keys, 1);
    assert_eq!(initial.compiled_keys, 1);
    assert_eq!(initial.cache_hits, 0);
    assert_eq!(initial.retained_keys, 1);
    let repeated = assessor.prewarm_prepared_graph(&borrowed).unwrap();
    assert_eq!(repeated.requested_keys, 1);
    assert_eq!(repeated.compiled_keys, 0);
    assert_eq!(repeated.cache_hits, 1);
    assert_eq!(repeated.retained_keys, 1);
    assert!(assessor.state().rocblas.get().is_none());
    for phase in [0_i16, 3, -2] {
        let left = (0..195)
            .map(|index| T::small(i16::try_from(index % 7).unwrap() + phase - 3))
            .collect::<Vec<_>>();
        let right = [T::small(1), T::small(-2), T::small(3)];
        assert_eq!(
            case.parity(assessor, session, &left, &right).unwrap().len(),
            65
        );
    }
    let mut faulting = vec![T::small(1); 195];
    faulting[64 * 3 + 2] = T::max();
    case.fault(
        &case
            .parity(assessor, session, &faulting, &[T::small(2); 3])
            .unwrap_err(),
        64,
        2,
        TensorArithmeticStep::Multiply,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    case.parity(assessor, session, &[T::small(1); 195], &[T::small(2); 3])
        .unwrap();
    let after_execution = assessor.prewarm_prepared_graph(&borrowed).unwrap();
    assert_eq!(after_execution.compiled_keys, 0);
    assert_eq!(after_execution.cache_hits, 1);
    assert_eq!(after_execution.retained_keys, 1);
    assert!(assessor.state().rocblas.get().is_none());
}

#[test]
#[ignore = "requires a working ROCm device; run serially"]
fn strict_matmul_f32_reference_transposes_partial_block_fault_retry_dependency() {
    let (_discovery, session) = crate::tensor::tests::rocm_test_session();
    let assessor = RocmTensorAssessor::new(&session).unwrap();
    transposes::<f32>(&assessor, &session);
    ordered_faults::<f32>(&assessor, &session);
    partial_block::<f32>(&assessor, &session);
}

#[test]
#[ignore = "requires a working ROCm device with FP64; run serially"]
fn strict_matmul_f64_reference_transposes_partial_block_fault_retry_dependency() {
    let (_discovery, session) = crate::tensor::tests::rocm_test_session();
    let assessor = RocmTensorAssessor::new(&session).unwrap();
    transposes::<f64>(&assessor, &session);
    ordered_faults::<f64>(&assessor, &session);
    partial_block::<f64>(&assessor, &session);
}

#[test]
#[ignore = "requires a working ROCm device; run serially"]
fn strict_matmul_many_typed_inputs_outputs_escape_original_owners() {
    let (_discovery, session) = crate::tensor::tests::rocm_test_session();
    let assessor = RocmTensorAssessor::new(&session).unwrap();
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let inputs: [ValueId; 9] =
        core::array::from_fn(|_| graph.input([1, 1], PcuScalarType::F32).unwrap());
    let product = graph.matmul(inputs[0], inputs[1]).unwrap();
    let sum = graph.add(inputs[2], inputs[3]).unwrap();
    // Requested order deliberately interleaves computed and identity transport outputs.
    // Every distinct input participates, exceeding the typed input adapter's inline capacity.
    let requested = [
        inputs[8], product, inputs[4], sum, inputs[7], inputs[5], inputs[6], inputs[0], inputs[1],
    ];
    let expected = [9.0_f32, 2.0, 5.0, 7.0, 8.0, 6.0, 7.0, 1.0, 2.0];
    let program = graph
        .into_selected_program(
            &requested,
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let prepared = assessor.prepare_owned_program(program).unwrap();
    let pool = PcuMemoryPoolId(0x534d_414e);
    let outputs = {
        let owners: Vec<_> = (1_i16..=9)
            .map(|value| {
                PcuDeviceTensor::new(
                    [1, 1],
                    session.upload_buffer(pool, &[f32::from(value)]).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let bindings: Vec<_> = inputs.iter().copied().zip(&owners).collect();
        let mut memory = session.memory_provider(pool);
        assessor
            .execute_owned_program_outputs(&prepared, &bindings, pool, &mut memory)
            .unwrap()
        // Input owners and provider adapter leave scope before any output is read.
    };
    drop(prepared);
    drop(assessor);
    assert_eq!(outputs.len(), requested.len());
    for ((actual_id, tensor), (expected_id, expected_value)) in
        outputs.iter().zip(requested.into_iter().zip(expected))
    {
        assert_eq!(*actual_id, expected_id);
        assert_eq!(tensor.shape(), [1, 1]);
        let mut observed = [0.0_f32];
        session
            .download_buffer(pool, tensor.buffer(), &mut observed)
            .unwrap();
        assert_eq!(observed[0].to_bits(), expected_value.to_bits());
    }
}

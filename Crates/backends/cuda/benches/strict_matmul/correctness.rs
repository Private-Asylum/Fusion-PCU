//! Untimed numerical preflights and reusable fault/retry controls.
#[rustfmt::skip]
use std::{
    error::Error,
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceTensor,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuMemoryPoolId,
    PcuNumericalMode,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    Tensor,
    TensorValue,
    TensorElement,
    TensorError,
    TensorArithmeticStep,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaOwnedDispatchBackend,
    CudaOwnedTensorAssessor,
    CudaTensorExecutionError,
    CudaRuntime,
};
#[rustfmt::skip]
use super::{
    native::Native,
    oracle::{self, Scalar},
    source,
};

fn source_fault<'a>(error: &'a (dyn Error + 'static)) -> Option<&'a TensorError> {
    if let Some(CudaTensorExecutionError::Graph(error)) =
        error.downcast_ref::<CudaTensorExecutionError>()
    {
        return Some(error);
    }
    error.source().and_then(source_fault)
}

pub fn boundary_rejected(error: &(dyn Error + 'static)) -> bool {
    if matches!(
        error.downcast_ref::<CudaTensorExecutionError>(),
        Some(CudaTensorExecutionError::Unsupported { .. })
    ) {
        return true;
    }
    error.source().is_some_and(boundary_rejected)
}

fn coordinates(error: &TensorError) -> (usize, usize, TensorArithmeticStep, PcuExecutionFaultKind) {
    match *error {
        TensorError::CompoundArithmeticFault {
            element_index,
            reduction_index,
            step,
            kind,
            ..
        } => (element_index, reduction_index, step, kind),
        _ => panic!("expected strict compound fault: {error:?}"),
    }
}

fn packed_fault(error: &TensorError) -> u64 {
    let (cell, depth, step, kind) = coordinates(error);
    let kind = match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 3,
        PcuExecutionFaultKind::ArithmeticUnderflow => 4,
        PcuExecutionFaultKind::InvalidFloatingOperand => 5,
        _ => panic!("unexpected MatMul fault"),
    };
    let step = u64::from(step == TensorArithmeticStep::Add);
    ((u64::try_from(cell * 3 + depth).unwrap() * 2 + step) << 3) | kind
}

#[allow(clippy::too_many_lines)] // Every exceptional case is compared across all three strict routes.
pub fn run<T: Scalar + TensorElement>(
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
) -> Result<(), Box<dyn Error>> {
    let zero = T::default();
    let one = T::small(1);
    let half = one.pcu_checked_div(T::small(2)).unwrap();
    let higher = one.pcu_checked_add(T::epsilon()).unwrap();
    let lower = one.pcu_checked_sub(T::epsilon()).unwrap();
    let ieee = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let cases = [
        (
            "intermediate_underflow",
            [T::min_subnormal(), one, zero],
            [half, one, one],
            ieee,
        ),
        (
            "final_underflow",
            [zero, zero, T::min_subnormal()],
            [one, one, half],
            ieee,
        ),
        (
            "multiply_overflow",
            [T::max(), zero, zero],
            [T::small(2), one, one],
            ieee,
        ),
        (
            "overflow_before_cancellation",
            [T::max(); 3],
            [one, one, T::small(-1)],
            ieee,
        ),
        (
            "final_add_overflow",
            [T::max(), zero, T::max()],
            [one; 3],
            ieee,
        ),
        (
            "exact_subnormal",
            [T::min_subnormal(), zero, zero],
            [one; 3],
            ieee,
        ),
        (
            "reject_exact_subnormal",
            [T::min_subnormal(), zero, zero],
            [one; 3],
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ),
        (
            "allow_gradual_intermediate",
            [T::min_subnormal(), one, zero],
            [half, one, one],
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ),
        (
            "exact_cancellation",
            [one, T::small(-1), zero],
            [one; 3],
            ieee,
        ),
        (
            "no_fma",
            [T::small(-1), higher, zero],
            [one, lower, one],
            ieee,
        ),
        (
            "invalid_operand",
            [T::infinity(), zero, zero],
            [zero; 3],
            ieee,
        ),
    ];
    let pool = PcuMemoryPoolId(0);
    for (label, left, right, policy) in cases {
        let left = [left];
        let right = right.map(|value| [value]);
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let left_id = graph.input([1, 3], T::TYPE)?;
        let right_id = graph.input([3, 1], T::TYPE)?;
        let output_id = graph.matmul(left_id, right_id)?;
        graph.set_value_float_underflow_policy(output_id, policy)?;
        let expected = graph.evaluate_checked(&[
            (
                left_id,
                TensorValue::from_tensor(Tensor::new([1, 3], left.as_flattened().to_vec())?),
            ),
            (
                right_id,
                TensorValue::from_tensor(Tensor::new([3, 1], right.as_flattened().to_vec())?),
            ),
        ]);
        let mut native = Native::new::<T, 1, 3, 1>(runtime, &graph, output_id)?;
        let program = graph.into_selected_program(
            &[output_id],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )?;
        let root = CudaOwnedTensorAssessor::new(Rc::clone(backend))?;
        let assessor = root.assessor();
        let prepared = assessor.prepare_owned_program(program)?;
        let mut memory = backend.memory_provider(pool);
        let source_result = match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => {
                source::product::<T, 1, 3, 1>(&left, &right)
            }
            PcuFloatUnderflowPolicy::RejectSubnormalResult => {
                source::reject_subnormal::<T, 1, 3, 1>(&left, &right)
            }
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => {
                source::gradual::<T, 1, 3, 1>(&left, &right)
            }
        };
        let raw_left =
            PcuDeviceTensor::new([1, 3], backend.upload_buffer(pool, left.as_flattened())?)?;
        let raw_right =
            PcuDeviceTensor::new([3, 1], backend.upload_buffer(pool, right.as_flattened())?)?;
        let graph_result = assessor.execute_owned_program_outputs(
            &prepared,
            &[(left_id, &raw_left), (right_id, &raw_right)],
            pool,
            &mut memory,
        );
        native.upload(left.as_flattened(), right.as_flattened())?;
        let native_result = native.submit()?;
        let mut observed = [zero];
        match expected {
            Ok(expected) => {
                let expected = expected.value_typed::<T>(output_id)?.data();
                source_result?.read_into(&mut observed)?;
                oracle::verify(expected, &observed);
                backend.download_buffer(pool, graph_result?[0].1.buffer(), &mut observed)?;
                oracle::verify(expected, &observed);
                assert_eq!(native_result, u64::MAX, "{label}");
                native.read(&mut observed)?;
                oracle::verify(expected, &observed);
            }
            Err(expected) => {
                let Err(source_error) = source_result else {
                    panic!("{label} source should fault")
                };
                assert_eq!(
                    coordinates(source_fault(&source_error).unwrap_or_else(|| panic!(
                        "{label} unexpected source error {source_error:?}"
                    ))),
                    coordinates(&expected)
                );
                let Err(CudaTensorExecutionError::Graph(graph_error)) = graph_result else {
                    panic!("{label} graph should report compound fault")
                };
                assert_eq!(coordinates(&graph_error), coordinates(&expected));
                assert_eq!(native_result, packed_fault(&expected), "{label}");
            }
        }
        // Reuse the same source specialization, prepared graph, native output/status after a fault.
        let safe_left = [[one; 3]];
        let safe_right = [[one]; 3];
        let retry = match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => {
                source::product::<T, 1, 3, 1>(&safe_left, &safe_right)?
            }
            PcuFloatUnderflowPolicy::RejectSubnormalResult => {
                source::reject_subnormal::<T, 1, 3, 1>(&safe_left, &safe_right)?
            }
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => {
                source::gradual::<T, 1, 3, 1>(&safe_left, &safe_right)?
            }
        };
        retry.read_into(&mut observed)?;
        oracle::verify(&[T::small(3)], &observed);
        let retry_left = PcuDeviceTensor::new(
            [1, 3],
            backend.upload_buffer(pool, safe_left.as_flattened())?,
        )?;
        let retry_right = PcuDeviceTensor::new(
            [3, 1],
            backend.upload_buffer(pool, safe_right.as_flattened())?,
        )?;
        let retried = assessor.execute_owned_program_outputs(
            &prepared,
            &[(left_id, &retry_left), (right_id, &retry_right)],
            pool,
            &mut memory,
        )?;
        backend.download_buffer(pool, retried[0].1.buffer(), &mut observed)?;
        oracle::verify(&[T::small(3)], &observed);
        native.upload(safe_left.as_flattened(), safe_right.as_flattened())?;
        assert_eq!(native.submit()?, u64::MAX, "{label} retry");
        native.read(&mut observed)?;
        oracle::verify(&[T::small(3)], &observed);
    }
    eprintln!(
        "preflight/{}: 11 numerical cases x source/explicit/native with success retry passed",
        T::LABEL
    );
    Ok(())
}

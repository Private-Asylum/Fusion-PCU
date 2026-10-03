//! Exact ordered loss step faults compared through source, graph, core and private native status.
#[rustfmt::skip]
use std::{error::Error, rc::Rc};
#[rustfmt::skip]
use fusion_pcu::{PcuDeviceTensor, PcuExecutionFaultKind, PcuFloatUnderflowPolicy, PcuMemoryPoolId, PcuNumericalMode, PcuNumericalOptions};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{Graph, Tensor, TensorValue, TensorElement, TensorError, TensorArithmeticStep, TensorArithmeticCapability, TensorArithmeticRewritePolicy, TensorPointwiseGroupingPolicy};
#[rustfmt::skip]
use fusion_pcu_cuda::{CudaOwnedDispatchBackend, CudaOwnedTensorAssessor, CudaRuntime, CudaTensorExecutionError};
#[rustfmt::skip]
use super::{native::Native, oracle::{self, Scalar}, source};
fn source_fault<'a>(error: &'a (dyn Error + 'static)) -> Option<&'a TensorError> {
    if let Some(CudaTensorExecutionError::Graph(error)) =
        error.downcast_ref::<CudaTensorExecutionError>()
    {
        return Some(error);
    }
    error
        .downcast_ref::<TensorError>()
        .or_else(|| error.source().and_then(source_fault))
}
fn coordinates(error: &TensorError) -> (usize, TensorArithmeticStep, PcuExecutionFaultKind) {
    let TensorError::CompoundArithmeticFault {
        element_index: 0,
        reduction_index,
        step,
        kind,
        ..
    } = *error
    else {
        panic!("ordered MSE compound fault: {error:?}");
    };
    (reduction_index, step, kind)
}
fn packed(error: &TensorError) -> u64 {
    let (index, step, kind) = coordinates(error);
    let event = u64::try_from(index).unwrap() * 3
        + match step {
            TensorArithmeticStep::Subtract | TensorArithmeticStep::Divide => 0,
            TensorArithmeticStep::Multiply => 1,
            TensorArithmeticStep::Add => 2,
        };
    let tag = match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 3,
        PcuExecutionFaultKind::ArithmeticUnderflow => 4,
        PcuExecutionFaultKind::InvalidFloatingOperand => 5,
        _ => panic!("ordered MSE fault kind"),
    };
    (event << 3) | tag
}
#[allow(clippy::too_many_lines)] // Every output/status/fault step and retry is compared at matching boundaries.
pub fn run<T: Scalar + TensorElement>(
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
    options: PcuNumericalOptions,
) -> Result<(), Box<dyn Error>> {
    let zero = T::default();
    let one = T::from_units(1, 1);
    for (label, prediction, policy) in [
        (
            "square_overflow",
            [one, T::max(), one],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "ordered_sum_overflow",
            [T::large_square_root(); 3],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "invalid_subtract",
            [one, T::infinity(), one],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "tiny_inexact_square",
            [T::tiny(), one, one],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "final_divide_tiny_inexact",
            [T::subnormal_square_root(), zero, zero],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "gradual_tiny_square",
            [T::tiny(); 3],
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ),
        (
            "tight_subtract",
            [T::tiny(), one, one],
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ),
        (
            "tight_exact_square",
            [T::subnormal_square_root(), one, one],
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ),
    ] {
        let target = [zero; 3];
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        graph.set_numerical_options(options);
        let x = graph.input([3], T::TYPE)?;
        let y = graph.input([3], T::TYPE)?;
        let output = graph.mean_squared_error(x, y)?;
        graph.set_value_float_underflow_policy(output, policy)?;
        let expected = graph.evaluate_checked(&[
            (
                x,
                TensorValue::from_tensor(Tensor::new([3], prediction.to_vec())?),
            ),
            (
                y,
                TensorValue::from_tensor(Tensor::new([3], target.to_vec())?),
            ),
        ]);
        let mut native = Native::new::<T, 3>(runtime, &graph, output)?;
        let root = CudaOwnedTensorAssessor::new(Rc::clone(backend))?;
        let assessor = root.assessor();
        let prepared = assessor.prepare_owned_program(graph.into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )?)?;
        let pool = PcuMemoryPoolId(0x4d53_4550);
        let mut memory = backend.memory_provider(pool);
        let x_owner = PcuDeviceTensor::new([3], backend.upload_buffer(pool, &prediction)?)?;
        let y_owner = PcuDeviceTensor::new([3], backend.upload_buffer(pool, &target)?)?;
        let explicit = assessor.execute_owned_program_outputs(
            &prepared,
            &[(x, &x_owner), (y, &y_owner)],
            pool,
            &mut memory,
        );
        let authored = match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => source::strict(&prediction, &target),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => source::gradual(&prediction, &target),
            PcuFloatUnderflowPolicy::RejectSubnormalResult => source::tight(&prediction, &target),
        };
        native.upload(0, &prediction, &target)?;
        let (native_output, status) = native.submit(0)?;
        let mut actual = [zero];
        match expected {
            Ok(expected) => {
                let expected = expected.value_typed::<T>(output)?.data();
                authored?.read_into(&mut actual)?;
                oracle::verify(expected, &actual);
                backend.download_buffer(pool, explicit?[0].1.buffer(), &mut actual)?;
                oracle::verify(expected, &actual);
                assert_eq!(status, u64::MAX);
                Native::read(&native_output, &mut actual)?;
                oracle::verify(expected, &actual);
            }
            Err(expected) => {
                let authored = authored.expect_err("authored compound fault");
                assert_eq!(
                    coordinates(source_fault(&authored).expect("source compound fault")),
                    coordinates(&expected),
                    "{label}"
                );
                let Err(CudaTensorExecutionError::Graph(explicit)) = explicit else {
                    panic!("explicit compound fault");
                };
                assert_eq!(coordinates(&explicit), coordinates(&expected), "{label}");
                assert_eq!(status, packed(&expected), "{label}");
            }
        }
        drop(native_output);
        let safe = [one; 3];
        source::strict(&safe, &target)?.read_into(&mut actual)?;
        oracle::verify(&[one], &actual);
        let safe_owner = PcuDeviceTensor::new([3], backend.upload_buffer(pool, &safe)?)?;
        let outputs = assessor.execute_owned_program_outputs(
            &prepared,
            &[(x, &safe_owner), (y, &y_owner)],
            pool,
            &mut memory,
        )?;
        backend.download_buffer(pool, outputs[0].1.buffer(), &mut actual)?;
        oracle::verify(&[one], &actual);
        native.upload(0, &safe, &target)?;
        let (output, status) = native.submit(0)?;
        assert_eq!(status, u64::MAX);
        Native::read(&output, &mut actual)?;
        oracle::verify(&[one], &actual);
        eprintln!(
            "correctness/{}/{options:?}/{label}: source+graph+core+native full output/status/ordered step and retry passed",
            T::LABEL
        );
    }
    Ok(())
}

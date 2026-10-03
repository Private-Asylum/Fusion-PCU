//! Independent core reference fault coordinates agree with source, graph and raw CUDA status.
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
    CudaRuntime,
    CudaTensorExecutionError,
};
#[rustfmt::skip]
use super::{
    native::Native,
    oracle::{
        self,
        Scalar,
    },
    source,
};
fn coordinates(error: &TensorError) -> (usize, TensorArithmeticStep, PcuExecutionFaultKind) {
    if let TensorError::CompoundArithmeticFault {
        element_index,
        reduction_index: 0,
        step,
        kind,
        ..
    } = *error
    {
        (element_index, step, kind)
    } else {
        panic!("compound SGD fault: {error:?}")
    }
}
fn source_fault<'a>(error: &'a (dyn Error + 'static)) -> Option<&'a TensorError> {
    error
        .downcast_ref::<TensorError>()
        .or_else(|| error.source().and_then(source_fault))
}
fn packed(error: &TensorError) -> u64 {
    let (index, step, kind) = coordinates(error);
    let tag = match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 3,
        PcuExecutionFaultKind::ArithmeticUnderflow => 4,
        PcuExecutionFaultKind::InvalidFloatingOperand => 5,
        _ => panic!("SGD float fault"),
    };
    ((u64::try_from(index).unwrap() * 2 + u64::from(step == TensorArithmeticStep::Subtract)) << 3)
        | tag
}
#[allow(clippy::too_many_lines)] // Complete exceptional outputs and terminal retry compare all three routes.
pub fn run<T: Scalar + TensorElement>(
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
    options: fusion_pcu::PcuNumericalOptions,
) -> Result<(), Box<dyn Error>> {
    let one = T::from_units(1, 1);
    let zero = T::default();
    let negative_max = zero.pcu_checked_sub(T::max()).unwrap();
    for (label, weights, gradient, policy) in [
        (
            "tiny_inexact_multiply",
            [one, one],
            [T::tiny(), one],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "subtract_overflow",
            [T::max(), one],
            [negative_max, one],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "invalid_weight_subtract_before_later_invalid_multiply",
            [T::infinity(), one],
            [one, T::infinity()],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "invalid_gradient_multiply",
            [one, one],
            [T::infinity(), one],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "exact_tiny_allowed",
            [T::tiny(), one],
            [zero, one],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        (
            "exact_tiny_rejected",
            [T::tiny(), one],
            [zero, one],
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ),
        (
            "tiny_inexact_gradual",
            [one, one],
            [T::tiny(), one],
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ),
    ] {
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        graph.set_numerical_options(options);
        let w = graph.input([2], T::TYPE)?;
        let g = graph.input([2], T::TYPE)?;
        let output = graph.sgd_update(w, g, 0.5)?;
        graph.set_value_float_underflow_policy(output, policy)?;
        let expected = graph.evaluate_checked(&[
            (
                w,
                TensorValue::from_tensor(Tensor::new([2], weights.to_vec())?),
            ),
            (
                g,
                TensorValue::from_tensor(Tensor::new([2], gradient.to_vec())?),
            ),
        ]);
        let mut native = Native::new::<T, 2>(runtime, &graph, output)?;
        let assessor_root = CudaOwnedTensorAssessor::new(Rc::clone(backend))?;
        let assessor = assessor_root.assessor();
        let prepared = assessor.prepare_owned_program(graph.into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )?)?;
        let pool = PcuMemoryPoolId(0x4353_4744);
        let mut memory = backend.memory_provider(pool);
        let w_owner = PcuDeviceTensor::new([2], backend.upload_buffer(pool, &weights)?)?;
        let g_owner = PcuDeviceTensor::new([2], backend.upload_buffer(pool, &gradient)?)?;
        let explicit = assessor.execute_owned_program_outputs(
            &prepared,
            &[(w, &w_owner), (g, &g_owner)],
            pool,
            &mut memory,
        );
        let authored = match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => source::update(&weights, &gradient),
            PcuFloatUnderflowPolicy::RejectSubnormalResult => source::tight(&weights, &gradient),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => source::gradual(&weights, &gradient),
        };
        native.upload(0, &weights, &gradient)?;
        let (native_output, status) = native.submit(0)?;
        let mut actual = [zero; 2];
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
                let authored = authored.expect_err("authored fault");
                assert_eq!(
                    coordinates(source_fault(&authored).expect("source compound fault")),
                    coordinates(&expected),
                    "{label}"
                );
                let Err(CudaTensorExecutionError::Graph(explicit)) = explicit else {
                    panic!("explicit compound fault")
                };
                assert_eq!(coordinates(&explicit), coordinates(&expected), "{label}");
                assert_eq!(status, packed(&expected), "{label}");
            }
        }
        drop(native_output);
        let safe = [one; 2];
        let expected = [T::from_units(1, 2); 2];
        source::update(&safe, &safe)?.read_into(&mut actual)?;
        oracle::verify(&expected, &actual);
        let safe_w = PcuDeviceTensor::new([2], backend.upload_buffer(pool, &safe)?)?;
        let safe_g = PcuDeviceTensor::new([2], backend.upload_buffer(pool, &safe)?)?;
        let outputs = assessor.execute_owned_program_outputs(
            &prepared,
            &[(w, &safe_w), (g, &safe_g)],
            pool,
            &mut memory,
        )?;
        backend.download_buffer(pool, outputs[0].1.buffer(), &mut actual)?;
        oracle::verify(&expected, &actual);
        native.upload(0, &safe, &safe)?;
        let (native_output, status) = native.submit(0)?;
        assert_eq!(status, u64::MAX);
        Native::read(&native_output, &mut actual)?;
        oracle::verify(&expected, &actual);
        eprintln!(
            "correctness/{}/{label}: requested={options:?} source+graph+native complete output/fault coordinates and terminal retry passed",
            T::LABEL
        );
    }
    Ok(())
}

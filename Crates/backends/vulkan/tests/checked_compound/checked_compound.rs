//! Native ordered compound geometry, source tuples and exact step/lane/reduction failures.
extern crate pcu_facade as fusion_pcu;
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../../benches/checked_compound/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/checked_compound/source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
    PcuExecutionFaultKind,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{
    Graph,
    TensorError,
    TensorArithmeticStep,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanPreparedTensorGraph,
    PcuVulkanTensorInput,
    PcuVulkanTensorError,
};
fn read<T: oracle::Format>(owner: &pcu_facade::PcuTensor<T>, expected: &[T]) {
    let sentinel = T::value(117.0);
    let mut actual = [sentinel; 7];
    owner.read_into(&mut actual).unwrap();
    for (a, b) in actual[..expected.len()].iter().zip(expected) {
        assert_eq!(a.bits(), b.bits());
    }
    for value in &actual[expected.len()..] {
        assert_eq!(value.bits(), sentinel.bits());
    }
}
fn source_profiles<T: oracle::Format>() {
    let oracle::Bank {
        left: a,
        right: b,
        weights: x,
        gradient: y,
        product,
        update,
        loss,
    } = oracle::banks::<T>(0);
    let output = source::product(&a, &b).unwrap();
    read(&output, &product);
    read(
        &source::chain(&a, &b, &y).unwrap(),
        &[-1.5, -6.0, -0.5, 8.5].map(T::value),
    );
    let mut invalid_target = y;
    invalid_target[1][0] = T::value(f32::NAN);
    assert_eq!(
        source::chain(&a, &b, &invalid_target)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap()
            .kind,
        PcuExecutionFaultKind::InvalidFloatingOperand
    );
    let left = source::identity(&x).unwrap();
    let right = source::identity(&y).unwrap();
    read(&source::loss(&left, &right).unwrap(), &[loss]);
    read(&source::loss(&left, &y).unwrap(), &[loss]);
    let updated = source::update(&left, &right).unwrap();
    read(&updated, &update);
    global::clear_thread_cache().unwrap();
    drop(left);
    drop(right);
    read(&updated, &update);
    let oracle::Bank {
        left: a,
        right: b,
        product,
        ..
    } = oracle::banks::<T>(1);
    read(&source::product(&a, &b).unwrap(), &product);
    read(&output, &oracle::banks::<T>(0).product);
}
#[test]
#[ignore = "actual Vulkan source/owner ordered compound permission proof"]
fn two_formats_all_strict_permissions_and_underflow_policies() {
    for compound_arithmetic in [
        PcuCompoundArithmeticPolicy::Checked,
        PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            for float_underflow in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                global::configure(global::PcuExecutionPolicy {
                    backend: global::PcuBackendChoice::Vulkan,
                    numerical_options: PcuNumericalOptions {
                        compound_arithmetic,
                        precision,
                        ..Default::default()
                    },
                    float_underflow,
                    ..Default::default()
                })
                .unwrap();
                source_profiles::<f32>();
                source_profiles::<f64>();
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
fn native_faults<T: oracle::Format>() {
    let (backend, _) = device::selected();
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([1, 2], T::TYPE).unwrap();
    let right = graph.input([2, 1], T::TYPE).unwrap();
    let output = graph.matmul(left, right).unwrap();
    let mut plan = PcuVulkanPreparedTensorGraph::<T>::prepare(&backend, &graph, &[output]).unwrap();
    drop(graph);
    let one = T::value(1.0);
    let half = T::value(0.5);
    let previous = plan
        .execute_owned(&[
            PcuVulkanTensorInput::Host(&[one, one]),
            PcuVulkanTensorInput::Host(&[one, one]),
        ])
        .unwrap();
    let inputs = [T::minimum(), one];
    let gradients = [half, one];
    assert!(
        matches!(plan.execute_owned(&[PcuVulkanTensorInput::Host(&inputs),PcuVulkanTensorInput::Host(&gradients)]),Err(PcuVulkanTensorError::Graph(TensorError::CompoundArithmeticFault {value,element_index:0,reduction_index:0,step:TensorArithmeticStep::Multiply,kind:PcuExecutionFaultKind::ArithmeticUnderflow})) if value==output)
    );
    let maximum = T::maximum();
    assert!(
        matches!(plan.execute_owned(&[PcuVulkanTensorInput::Host(&[maximum,maximum]),PcuVulkanTensorInput::Host(&[one,one])]),Err(PcuVulkanTensorError::Graph(TensorError::CompoundArithmeticFault {value,element_index:0,reduction_index:1,step:TensorArithmeticStep::Add,kind:PcuExecutionFaultKind::ArithmeticOverflow})) if value==output)
    );
    let mut observed = [T::value(117.0); 3];
    previous.read_into(&mut observed).unwrap();
    assert_eq!(observed[0].bits(), T::value(2.0).bits());
    let owner = backend.upload_owned(&[one, one]).unwrap();
    plan.execute_owned(&[
        PcuVulkanTensorInput::Owned(&owner),
        PcuVulkanTensorInput::Host(&[one, one]),
    ])
    .unwrap()
    .read_into(&mut observed)
    .unwrap();
    assert_eq!(observed[0].bits(), T::value(2.0).bits());
    assert_eq!(observed[1].bits(), T::value(117.0).bits());
}
#[test]
#[ignore = "actual unequal native extents and intermediate fault publication"]
fn native_reduction_step_provenance_and_private_retry() {
    native_faults::<f32>();
    native_faults::<f64>();
}

fn transposes<T: oracle::Format>() {
    let (backend, _) = device::selected();
    let bank = oracle::banks::<T>(1);
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        for transpose_left in [false, true] {
            for transpose_right in [false, true] {
                let mut graph = Graph::default();
                graph.set_numerical_mode(PcuNumericalMode::Strict);
                let left = graph
                    .input(if transpose_left { [3, 2] } else { [2, 3] }, T::TYPE)
                    .unwrap();
                let right = graph
                    .input(if transpose_right { [2, 3] } else { [3, 2] }, T::TYPE)
                    .unwrap();
                let output = graph
                    .matmul_transposed(left, right, transpose_left, transpose_right)
                    .unwrap();
                graph
                    .set_value_float_underflow_policy(output, policy)
                    .unwrap();
                let mut plan =
                    PcuVulkanPreparedTensorGraph::<T>::prepare(&backend, &graph, &[output])
                        .unwrap();
                let a: Vec<_> = (0..6)
                    .map(|i| {
                        if transpose_left {
                            bank.left[i % 2][i / 2]
                        } else {
                            bank.left[i / 3][i % 3]
                        }
                    })
                    .collect();
                let b: Vec<_> = (0..6)
                    .map(|i| {
                        if transpose_right {
                            bank.right[i % 3][i / 3]
                        } else {
                            bank.right[i / 2][i % 2]
                        }
                    })
                    .collect();
                let ra = backend.upload_owned(&a).unwrap();
                let rb = backend.upload_owned(&b).unwrap();
                let result = plan
                    .execute_owned(&[
                        PcuVulkanTensorInput::Owned(&ra),
                        PcuVulkanTensorInput::Owned(&rb),
                    ])
                    .unwrap();
                drop(graph);
                drop(ra);
                drop(rb);
                let mut actual = [T::value(117.0); 7];
                result.read_into(&mut actual).unwrap();
                for (a, b) in actual[..4].iter().zip(&bank.product) {
                    assert_eq!(a.bits(), b.bits());
                }
                for tail in &actual[4..] {
                    assert_eq!(tail.bits(), T::value(117.0).bits());
                }
            }
        }
    }
}
#[test]
#[ignore = "actual matrix transpose flags use unequal retained source shapes"]
fn all_transposes_and_three_policies_are_exact() {
    transposes::<f32>();
    transposes::<f64>();
}

fn third_loss_and_empty_inner<T: oracle::Format>() {
    let (backend, _) = device::selected();
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let a = graph.input_typed::<T>([3]).unwrap();
    let b = graph.input_typed::<T>([3]).unwrap();
    let loss = graph.mean_squared_error_typed(a, b).unwrap().erase();
    let mut plan = PcuVulkanPreparedTensorGraph::<T>::prepare(&backend, &graph, &[loss]).unwrap();
    let mut observed = [T::value(117.0); 8];
    plan.execute_owned(&[
        PcuVulkanTensorInput::Host(&[1.0, 2.0, 3.0].map(T::value)),
        PcuVulkanTensorInput::Host(&[T::value(0.0); 3]),
    ])
    .unwrap()
    .read_into(&mut observed)
    .unwrap();
    // 1^2+2^2+3^2=14. Integer quotient/remainder nearest-even for14/3 gives
    // F32 0x40955555, F64 0x4012aaaaaaaaaaab, independently of the reference evaluator.
    assert_eq!(observed[0].bits(), T::third_loss().bits());
    for tail in &observed[1..] {
        assert_eq!(tail.bits(), T::value(117.0).bits());
    }
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let a = graph.input([2, 0], T::TYPE).unwrap();
    let b = graph.input([0, 3], T::TYPE).unwrap();
    let output = graph.matmul(a, b).unwrap();
    let mut plan = PcuVulkanPreparedTensorGraph::<T>::prepare(&backend, &graph, &[output]).unwrap();
    plan.execute_owned(&[
        PcuVulkanTensorInput::Host(&[]),
        PcuVulkanTensorInput::Host(&[]),
    ])
    .unwrap()
    .read_into(&mut observed)
    .unwrap();
    for value in &observed[..6] {
        assert_eq!(value.bits(), T::value(0.0).bits());
    }
    for tail in &observed[6..] {
        assert_eq!(tail.bits(), T::value(117.0).bits());
    }
}
#[test]
#[ignore = "actual non-power-of-two ordered MSE and zero-inner dot products"]
fn independently_rounded_thirds_and_empty_inner_have_exact_bits() {
    third_loss_and_empty_inner::<f32>();
    third_loss_and_empty_inner::<f64>();
}

fn faults_mse_sgd<T: oracle::Format>() {
    let (backend, _) = device::selected();
    let one = T::value(1.0);
    let maximum = T::maximum();
    let negative = T::value(-1.0);
    for operation in [0, 1] {
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let a = graph.input_typed::<T>([4]).unwrap();
        let b = graph.input_typed::<T>([4]).unwrap();
        let output = if operation == 0 {
            graph.mean_squared_error_typed(a, b).unwrap().erase()
        } else {
            graph.sgd_update(a.erase(), b.erase(), 1.0).unwrap()
        };
        let mut plan =
            PcuVulkanPreparedTensorGraph::<T>::prepare(&backend, &graph, &[output]).unwrap();
        let good = plan
            .execute_owned(&[
                PcuVulkanTensorInput::Host(&[one; 4]),
                PcuVulkanTensorInput::Host(&[one; 4]),
            ])
            .unwrap();
        let mut left = [one; 4];
        let mut right = [one; 4];
        left[2] = maximum;
        right[2] = maximum;
        if operation == 0 {
            right[2] = negative;
        } else {
            right[2] = T::value(-1.0);
            left[2] = T::value(-1.0);
        }
        if operation == 0 {
            assert!(
                matches!(plan.execute_owned(&[PcuVulkanTensorInput::Host(&left),PcuVulkanTensorInput::Host(&right)]),Err(PcuVulkanTensorError::Graph(TensorError::CompoundArithmeticFault{value,element_index:0,reduction_index:2,step:TensorArithmeticStep::Multiply,kind:PcuExecutionFaultKind::ArithmeticOverflow})) if value==output)
            );
        } else {
            // NaN weight is checked only after the independently checked gradient product.
            left[2] = T::value(f32::NAN);
            assert!(
                matches!(plan.execute_owned(&[PcuVulkanTensorInput::Host(&left),PcuVulkanTensorInput::Host(&right)]),Err(PcuVulkanTensorError::Graph(TensorError::CompoundArithmeticFault{value,element_index:2,reduction_index:0,step:TensorArithmeticStep::Subtract,kind:PcuExecutionFaultKind::InvalidFloatingOperand})) if value==output)
            );
        }
        let mut actual = [one; 5];
        good.read_into(&mut actual).unwrap();
        for value in &actual[..if operation == 0 { 1 } else { 4 }] {
            assert_eq!(value.bits(), T::value(0.0).bits());
        }
        plan.execute_owned(&[
            PcuVulkanTensorInput::Host(&[one; 4]),
            PcuVulkanTensorInput::Host(&[one; 4]),
        ])
        .unwrap();
    }
}
#[test]
#[ignore = "actual MSE and SGD named constituent fault records"]
fn loss_and_update_preserve_reduction_and_subtract_fault_steps() {
    faults_mse_sgd::<f32>();
    faults_mse_sgd::<f64>();
}

fn cold<T: oracle::Format>() {
    for compound_arithmetic in [
        PcuCompoundArithmeticPolicy::Checked,
        PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                let mut graph = Graph::default();
                graph.set_numerical_mode(PcuNumericalMode::Strict);
                graph.set_numerical_options(PcuNumericalOptions {
                    compound_arithmetic,
                    precision,
                    ..Default::default()
                });
                let a = graph.input_typed::<T>([2, 2]).unwrap();
                let b = graph.input_typed::<T>([2, 2]).unwrap();
                let product = graph.matmul(a.erase(), b.erase()).unwrap();
                let loss = graph.mean_squared_error_typed(a, b).unwrap().erase();
                let update = graph.sgd_update(a.erase(), b.erase(), 0.5).unwrap();
                for node in [product, loss, update] {
                    graph
                        .set_value_float_underflow_policy(node, policy)
                        .unwrap();
                    PcuVulkanPreparedTensorGraph::<T>::assess(&graph, &[node]).unwrap();
                }
            }
        }
    }
    let mut graph = Graph::default();
    let a = graph.input([2, 2], T::TYPE).unwrap();
    let out = graph.matmul(a, a).unwrap();
    assert!(matches!(
        PcuVulkanPreparedTensorGraph::<T>::assess(&graph, &[out]),
        Err(PcuVulkanTensorError::Graph(
            TensorError::UnsupportedNumericalMode { .. }
        ))
    ));
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    graph.set_numerical_options(PcuNumericalOptions {
        reproducibility: pcu_facade::PcuReproducibility::PortableV1,
        ..Default::default()
    });
    let out = graph.matmul(a, a).unwrap();
    assert!(PcuVulkanPreparedTensorGraph::<T>::assess(&graph, &[out]).is_err());
}
#[test]
fn exact_cold_strict_permissions_leave_boundary_and_portable_closed() {
    cold::<f32>();
    cold::<f64>();
}

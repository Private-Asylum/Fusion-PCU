//! Cold count, gradient-scale and learning-rate formation must use fixed nearest-even bits.
#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[path = "environment/environment.rs"]
mod environment;
#[rustfmt::skip]
use pcu_facade::{
    PcuNumericalMode,
    PcuScalarType,
    PcuFloatUnderflowPolicy,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{
    Graph,
    OpDescriptor,
    TensorScalarValue,
};
use fusion_pcu_cpu::PcuCpuPreparedTensorGraph;
use std::hint::black_box;

fn inspect(label: &str, actual: u64, expected: u64, failures: &mut usize) {
    if actual != expected {
        *failures += 1;
        eprintln!("cold mismatch {label}: actual={actual:016x} expected={expected:016x}");
    }
}
fn graph(
    scalar: PcuScalarType,
    count: usize,
    policy: PcuFloatUnderflowPolicy,
) -> (Graph, pcu_facade::dialect::tensor::ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let prediction = graph.input([black_box(count)], scalar).unwrap();
    let target = graph.input([count], scalar).unwrap();
    let loss = graph.mean_squared_error(prediction, target).unwrap();
    let derivative = graph.backward_mse_for(loss, prediction).unwrap();
    apply_policy(&mut graph, policy);
    (graph, derivative)
}
fn apply_policy(graph: &mut Graph, policy: PcuFloatUnderflowPolicy) {
    let values: Vec<_> = graph
        .nodes()
        .filter(|node| node.float_underflow_policy.is_some())
        .map(|node| node.value)
        .collect();
    for value in values {
        graph
            .set_value_float_underflow_policy(value, policy)
            .unwrap();
    }
}
fn scale(graph: &Graph) -> u64 {
    graph
        .nodes()
        .find_map(|node| match node.op {
            OpDescriptor::Uniform {
                value: TensorScalarValue::F32(value),
            } => Some(u64::from(value.to_bits())),
            OpDescriptor::Uniform {
                value: TensorScalarValue::F64(value),
            } => Some(value.to_bits()),
            _ => None,
        })
        .unwrap()
}
fn counts(failures: &mut usize) {
    // Metadata-only graphs prove rounding at precision boundaries without huge dense allocations.
    for (scalar, count, expected) in [
        (PcuScalarType::F32, (1_usize << 24) + 1, 0x3400_0000),
        (
            PcuScalarType::F64,
            (1_usize << 53) + 1,
            0x3cb0_0000_0000_0000,
        ),
    ] {
        let (graph, _) = graph(
            scalar,
            count,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        );
        inspect(
            "large-count gradient scale",
            scale(&graph),
            expected,
            failures,
        );
    }
}
fn gradients(policy: PcuFloatUnderflowPolicy, failures: &mut usize) {
    let (graph32, derivative32) = graph(PcuScalarType::F32, 3, policy);
    inspect("F32 cold 2/3", scale(&graph32), 0x3f2a_aaab, failures);
    let mut plan32 = PcuCpuPreparedTensorGraph::<f32>::prepare(&graph32, &[derivative32]).unwrap();
    let mut output32 = [7.0_f32; 5];
    plan32
        .call(&[&[1.0; 3], &[0.0; 3]], &mut [&mut output32])
        .unwrap();
    for value in &output32[..3] {
        inspect(
            "F32 prepared derivative",
            u64::from(value.to_bits()),
            0x3f2a_aaab,
            failures,
        );
    }
    assert_eq!(&output32[3..], &[7.0; 2]);
    let (graph64, derivative64) = graph(PcuScalarType::F64, 3, policy);
    inspect(
        "F64 cold 2/3",
        scale(&graph64),
        0x3fe5_5555_5555_5555,
        failures,
    );
    let mut plan64 = PcuCpuPreparedTensorGraph::<f64>::prepare(&graph64, &[derivative64]).unwrap();
    let mut output64 = [7.0_f64; 5];
    plan64
        .call(&[&[1.0; 3], &[0.0; 3]], &mut [&mut output64])
        .unwrap();
    for value in &output64[..3] {
        inspect(
            "F64 prepared derivative",
            value.to_bits(),
            0x3fe5_5555_5555_5555,
            failures,
        );
    }
    assert_eq!(&output64[3..], &[7.0; 2]);
}
fn rate(policy: PcuFloatUnderflowPolicy, failures: &mut usize) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let weights = graph.input([1], PcuScalarType::F64).unwrap();
    let upstream = graph.input([1], PcuScalarType::F64).unwrap();
    let update = graph
        .sgd_update(weights, upstream, black_box(f32::from_bits(black_box(1))))
        .unwrap();
    let target = graph.input([1], PcuScalarType::F64).unwrap();
    let loss = graph.mean_squared_error(update, target).unwrap();
    let derivative = graph.backward_mse_for(loss, upstream).unwrap();
    apply_policy(&mut graph, policy);
    let negative_rate = graph
        .nodes()
        .find_map(|node| match node.op {
            OpDescriptor::Uniform {
                value: TensorScalarValue::F64(value),
            } if value.to_bits() >> 63 != 0 => Some(value.to_bits()),
            _ => None,
        })
        .unwrap();
    inspect(
        "F64 cold negative subnormal F32 rate",
        negative_rate,
        0xb6a0_0000_0000_0000,
        failures,
    );
    let mut forward = PcuCpuPreparedTensorGraph::<f64>::prepare(&graph, &[update]).unwrap();
    let mut output = [7.0_f64; 3];
    forward
        .call(&[&[0.0], &[1.0], &[0.0]], &mut [&mut output])
        .unwrap();
    inspect(
        "F64 prepared forward rate",
        output[0].to_bits(),
        0xb6a0_0000_0000_0000,
        failures,
    );
    assert_eq!(&output[1..], &[7.0; 2]);
    let mut backward = PcuCpuPreparedTensorGraph::<f64>::prepare(&graph, &[derivative]).unwrap();
    backward
        .call(&[&[0.0], &[0.0], &[1.0]], &mut [&mut output])
        .unwrap();
    inspect(
        "F64 prepared backward rate",
        output[0].to_bits(),
        0x36b0_0000_0000_0000,
        failures,
    );
}
#[test]
fn cold_constants_ignore_directed_rounding_and_flush_inputs() {
    let original = environment::state();
    let mut failures = 0;
    for rounding in 0..4 {
        for flush in [false, true] {
            {
                let _guard = environment::Guard::enter(rounding, flush);
                eprintln!(
                    "cold state rounding={rounding} flush={flush} {:?}",
                    environment::state()
                );
                counts(&mut failures);
                for policy in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    gradients(policy, &mut failures);
                    rate(policy, &mut failures);
                }
            }
            assert_eq!(
                environment::state(),
                original,
                "current-thread state was not restored"
            );
        }
    }
    assert_eq!(
        failures, 0,
        "cold constants differ from independent nearest-even bit expectations"
    );
}

//! Criterion composition for a PCU tensor `ReLU` versus native HIP `ReLU`.

#[path = "support/elementwise.rs"]
mod elementwise;
mod support;

use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use fusion_pcu_tensor::{
    Graph,
    TensorError,
    ValueId,
};

use elementwise::ElementwiseCase;

const SOURCE: &str = r#"
extern "C" __global__ void native_relu(const float *input, float *output, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float value = input[id];
        output[id] = value > 0.0f ? value : 0.0f;
    }
}
"#;

fn values(size: usize) -> Vec<Vec<f32>> {
    vec![
        (0..size)
            .map(|id| if id.is_multiple_of(2) { -1.25 } else { 2.5 })
            .collect(),
    ]
}

fn expected(inputs: &[Vec<f32>]) -> Vec<f32> {
    inputs[0].iter().map(|value| value.max(0.0)).collect()
}

fn graph(graph: &mut Graph, inputs: &[ValueId]) -> Result<ValueId, TensorError> {
    graph.relu(inputs[0])
}

fn bench(criterion: &mut Criterion) {
    elementwise::run(
        criterion,
        &ElementwiseCase {
            name: "relu",
            source: SOURCE,
            kernel_name: c"native_relu",
            input_count: 1,
            values,
            expected,
            graph,
        },
    )
    .expect("paired ROCm ReLU benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = bench
}
criterion_main!(benches);

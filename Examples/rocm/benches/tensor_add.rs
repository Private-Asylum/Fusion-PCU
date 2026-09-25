//! Criterion composition for a PCU tensor Add versus native HIP Add.

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
extern "C" __global__ void native_add(const float *a, const float *b, float *out, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) out[id] = a[id] + b[id];
}
"#;

fn values(size: usize) -> Vec<Vec<f32>> {
    vec![vec![1.25; size], vec![2.5; size]]
}

fn expected(inputs: &[Vec<f32>]) -> Vec<f32> {
    inputs[0]
        .iter()
        .zip(&inputs[1])
        .map(|(left, right)| left + right)
        .collect()
}

fn graph(graph: &mut Graph, inputs: &[ValueId]) -> Result<ValueId, TensorError> {
    graph.add(inputs[0], inputs[1])
}

fn bench(criterion: &mut Criterion) {
    elementwise::run(
        criterion,
        &ElementwiseCase {
            name: "add",
            source: SOURCE,
            kernel_name: c"native_add",
            input_count: 2,
            values,
            expected,
            graph,
        },
    )
    .expect("paired ROCm Add benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = bench
}
criterion_main!(benches);

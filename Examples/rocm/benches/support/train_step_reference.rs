//! CPU reference evaluation and output checks for the training-step benchmark.

use std::error::Error;

use fusion_pcu_tensor::Tensor;

use super::{
    Program,
    StrictProgram,
};

pub fn cpu_strict_two_steps(
    program: &StrictProgram,
    samples: &Tensor,
    target: &Tensor,
    initial: &Tensor,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut weights = initial.clone();
    for _ in 0..2 {
        let inputs = [
            (program.samples, samples.clone()),
            (program.weights, weights.clone()),
            (program.target, target.clone()),
        ];
        let execution = program.graph.evaluate(&inputs)?;
        weights = execution.value(program.output)?.clone();
    }
    Ok(weights.data().to_vec())
}

pub fn cpu_two_steps(
    program: &Program,
    samples: &Tensor,
    target: &Tensor,
    rate: &Tensor,
    initial: &Tensor,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut weights = initial.clone();
    for _ in 0..2 {
        let inputs = [
            (program.samples, samples.clone()),
            (program.weights, weights.clone()),
            (program.target, target.clone()),
            (program.rate, rate.clone()),
        ];
        let execution = program.graph.evaluate(&inputs)?;
        weights = execution.value(program.output)?.clone();
    }
    Ok(weights.data().to_vec())
}

pub fn verify(expected: &[f32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    if expected.len() != actual.len() {
        return Err(format!(
            "training output length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        )
        .into());
    }
    let (index, error) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .map(|(index, (expected, actual))| (index, (expected - actual).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or("empty training output")?;
    if error <= 2.0e-4 {
        Ok(())
    } else {
        Err(format!(
            "training output mismatch: max abs error {error} at {index}: expected {}, got {}",
            expected[index], actual[index]
        )
        .into())
    }
}

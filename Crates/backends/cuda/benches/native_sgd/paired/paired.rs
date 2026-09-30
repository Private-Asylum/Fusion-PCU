//! Separate rotating-order paired diagnostic; this skips canonical Criterion primary sampling.
use std::time::Instant;
#[rustfmt::skip]
use super::{
    driver::HostCall,
    oracle::{
        self,
        Fixture,
    },
};

fn summary(samples: [f64; 20]) -> (f64, f64) {
    let mean = samples.iter().sum::<f64>() / 20.0;
    let variance = samples
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / 19.0;
    // Two-sided 95% Student interval with 19 degrees of freedom, block-level observations.
    let half = 2.093_024_054 * (variance / 20.0).sqrt();
    (mean, half)
}

pub fn run<const N: usize>(
    label: &str,
    fixtures: &[Fixture<N>; 2],
    calls: &mut [&mut HostCall<'_, N>; 3],
) {
    let wall = Instant::now();
    let mut samples = [[0.0; 20]; 3];
    let mut observed = vec![0.0; N];
    for block in 0..20 {
        for offset in 0..3 {
            let route = (block + offset) % 3;
            let mut elapsed = 0.0;
            for repeat in 0..8 {
                let input = &fixtures[(block + repeat) % 2];
                let start = Instant::now();
                calls[route](&input.weights, &input.gradient, &mut observed);
                elapsed = start.elapsed().as_secs_f64().mul_add(1e9, elapsed);
                oracle::verify(&input.expected, &observed);
            }
            samples[route][block] = elapsed / 8.0;
        }
        eprintln!(
            "paired/raw/{label}/block={block}/source_ns={}/graph_ns={}/native_ns={}",
            samples[0][block], samples[1][block], samples[2][block]
        );
    }
    for (route, values) in ["source", "graph", "native"].into_iter().zip(samples) {
        let (mean, half) = summary(values);
        eprintln!(
            "paired/summary/{label}/{route}/mean_ns={mean}/95CI_ns=[{},{}]",
            mean - half,
            mean + half
        );
    }
    for (route, values) in ["source_minus_native", "graph_minus_native"]
        .into_iter()
        .zip([samples[0], samples[1]])
    {
        let differences = core::array::from_fn(|index| values[index] - samples[2][index]);
        let (mean, half) = summary(differences);
        eprintln!(
            "paired/difference/{label}/{route}/mean_ns={mean}/95CI_ns=[{},{}]",
            mean - half,
            mean + half
        );
    }
    eprintln!(
        "paired/{label}/wall={:?};20blocks,8calls/route/block,rotatingorder; matched dynamic caller dispatch and Instant instrumentation diagnostic; no allocator counter, no general speedup claim",
        wall.elapsed()
    );
}

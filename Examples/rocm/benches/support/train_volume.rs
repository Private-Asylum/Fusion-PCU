//! Per-job payload generation and independent strict two-step reference for volume training.

use std::error::Error;

pub const ROWS: usize = 4;
pub const FEATURES: usize = 2;
pub const LEARNING_RATE: f32 = 0.001;
pub const MAX_JOB_ID: u64 = (1_u64 << 46) - 1;
const MANTISSA_MASK: u64 = (1_u64 << 23) - 1;

#[derive(Clone)]
pub struct JobInputs {
    pub id: u64,
    pub samples: Vec<f32>,
    pub target: Vec<f32>,
    pub initial_weights: Vec<f32>,
}

/// The two weights carry the low and high 23 ID bits as exact mantissa tags.
/// Both remain in [0.5, 1), so the tags do not create extreme inputs.
pub fn make_job(id: u64) -> Result<JobInputs, Box<dyn Error>> {
    if id > MAX_JOB_ID {
        return Err(format!("job ID {id} exceeds exact unique f32 range {MAX_JOB_ID}").into());
    }
    let low_tag = u32::try_from(id & MANTISSA_MASK)?;
    let high_tag = u32::try_from(id >> 23)?;
    let first_weight = f32::from_bits(0x3f00_0000 | low_tag);
    let second_weight = f32::from_bits(0x3f00_0000 | high_tag);
    let salt = id;
    let mix = |value: u64| {
        let mut value = value;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    };
    let sample_value = |index: usize| {
        let mixed = mix(salt
            ^ u64::try_from(index)
                .expect("fixed sample index fits u64")
                .wrapping_add(1)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15));
        (f32::from(u16::try_from(mixed % 2001).expect("bounded sample")) - 1000.0) / 1024.0
    };
    let target_value = |index: usize| {
        let mixed = mix(salt
            ^ u64::try_from(index)
                .expect("fixed target index fits u64")
                .wrapping_add(17)
                .wrapping_mul(0xd1b5_4a32_d192_ed03));
        (f32::from(u16::try_from(mixed % 2001).expect("bounded target")) - 1000.0) / 512.0
    };
    let initial_weights = vec![first_weight, second_weight];
    let samples = (0..ROWS * FEATURES).map(sample_value).collect();
    let target = (0..ROWS).map(target_value).collect();
    Ok(JobInputs {
        id,
        samples,
        target,
        initial_weights,
    })
}

/// Independent row-major CPU implementation matching two strict SGD steps.
#[allow(clippy::suboptimal_flops)] // Preserve separate f32 rounding; this is a strict reference.
pub fn cpu_two_steps(job: &JobInputs) -> Vec<f32> {
    let mut weights = job.initial_weights.clone();
    for _ in 0..2 {
        let mut gradient = [0.0_f32; FEATURES];
        for row in 0..ROWS {
            let prediction = job.samples[row * FEATURES] * weights[0]
                + job.samples[row * FEATURES + 1] * weights[1];
            let error = (prediction - job.target[row]) * 0.5;
            for (feature, value) in gradient.iter_mut().enumerate() {
                *value += job.samples[row * FEATURES + feature] * error;
            }
        }
        for (weight, sum) in weights.iter_mut().zip(gradient) {
            *weight -= LEARNING_RATE * sum;
        }
    }
    weights
}

pub fn verify(expected: &[f32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    if expected.len() != actual.len()
        || expected
            .iter()
            .chain(actual)
            .any(|value| !value.is_finite())
    {
        return Err("training output length mismatch or non-finite output".into());
    }
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let error = (expected - actual).abs();
        if error > 2.0e-4 {
            return Err(format!(
                "output mismatch at {index}: expected {expected}, got {actual} (error {error})"
            )
            .into());
        }
    }
    Ok(())
}

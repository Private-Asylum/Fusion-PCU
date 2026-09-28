//! Input freshness and reference checks for the volume benchmark, without GPU access.

#[path = "../benches/support/train_volume.rs"]
#[allow(dead_code)] // This test exercises input generation, not the benchmark execution helpers.
mod train_volume;

#[test]
fn job_identity_rolls_into_the_second_exact_mantissa_tag() {
    let before = train_volume::make_job((1_u64 << 23) - 1).expect("job before rollover");
    let after = train_volume::make_job(1_u64 << 23).expect("job after rollover");
    assert_eq!(before.initial_weights[0].to_bits() & 0x7f_ffff, 0x7f_ffff);
    assert_eq!(after.initial_weights[0].to_bits() & 0x7f_ffff, 0);
    assert_eq!(before.initial_weights[1].to_bits() & 0x7f_ffff, 0);
    assert_eq!(after.initial_weights[1].to_bits() & 0x7f_ffff, 1);
    assert!(
        before
            .initial_weights
            .iter()
            .chain(&after.initial_weights)
            .all(|w| { w.is_finite() && (0.5..1.0).contains(w) })
    );
}

#[test]
fn identity_pair_is_unique_through_maximum_id_and_exhaustion_is_rejected() {
    let ids = [
        0,
        1,
        (1_u64 << 23) - 1,
        1_u64 << 23,
        train_volume::MAX_JOB_ID - 1,
        train_volume::MAX_JOB_ID,
    ];
    let tags: Vec<_> = ids
        .into_iter()
        .map(|id| {
            let job = train_volume::make_job(id).expect("in-range job");
            (
                job.initial_weights[0].to_bits(),
                job.initial_weights[1].to_bits(),
            )
        })
        .collect();
    for left in 0..tags.len() {
        for right in left + 1..tags.len() {
            assert_ne!(tags[left], tags[right], "IDs {left} and {right} collided");
        }
    }
    assert!(train_volume::make_job(train_volume::MAX_JOB_ID + 1).is_err());
}

#[test]
fn verifier_rejects_a_previous_jobs_result_and_nonfinite_values() {
    let first = train_volume::make_job(0).expect("first job");
    let second = train_volume::make_job(train_volume::MAX_JOB_ID).expect("distant job");
    let old = train_volume::cpu_two_steps(&first);
    let expected = train_volume::cpu_two_steps(&second);
    assert!(train_volume::verify(&expected, &old).is_err());
    assert!(train_volume::verify(&expected, &expected).is_ok());
    assert!(train_volume::verify(&expected, &[f32::NAN, 0.0]).is_err());
    assert!(train_volume::verify(&expected, &[f32::INFINITY, 0.0]).is_err());
}

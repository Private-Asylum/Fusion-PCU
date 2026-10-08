//! Host-visible fault latency, separately from successful execution and payload readback.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuExecutionFaultKind,
    PcuExecutionObservationPolicy,
    PcuNumericalMode,
};
use criterion::Criterion;
#[rustfmt::skip]
use std::time::{
    Duration,
    Instant,
};

pub fn run(c: &mut Criterion, device: u32) {
    case::<65>(c, device);
    case::<4096>(c, device);
}

fn case<const N: usize>(c: &mut Criterion, device: u32) {
    let right = [1_u32; N];
    let factor = [2_u32; N];
    let mut input = [1_u32; N];
    for observation in [
        PcuExecutionObservationPolicy::Automatic,
        PcuExecutionObservationPolicy::HostObservedStages,
    ] {
        global::clear_thread_cache().unwrap();
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Rocm,
            device: Some(device),
            numerical_mode: PcuNumericalMode::Strict,
            observation,
            ..Default::default()
        })
        .unwrap();
        for (stage, bad) in [("first", u32::MAX), ("second", u32::MAX / 2)] {
            input[N - 1] = bad;
            // Resolve preparation, compilation, storage and the expected fault before timing.
            verify(
                &super::source::pipeline(&input, &right, &factor).unwrap_err(),
                N,
            );
            // Let the preceding measured group leave the utilization sampling window.
            // Cooldown is outside Criterion and every measured host-call interval.
            std::thread::sleep(Duration::from_secs(5));
            super::activity::activity_guard();
            {
                let mut group =
                    c.benchmark_group(format!("rocm_guarded_fault/{observation:?}/{stage}/{N}"));
                group.sample_size(20);
                group.warm_up_time(Duration::from_millis(500));
                group.measurement_time(Duration::from_secs(2));
                let mut generation = 0_u32;
                group.bench_function("host_error_boundary", |b| {
                    b.iter_custom(|iterations| {
                        let mut elapsed = Duration::ZERO;
                        for _ in 0..iterations {
                            // Distinct safe data rules out a result-cache explanation. Input edits
                            // and correctness assertions are outside the measured call boundary.
                            generation = generation.wrapping_add(1);
                            input[0] = generation % 1024;
                            let start = Instant::now();
                            let result = super::source::pipeline(&input, &right, &factor);
                            elapsed += start.elapsed();
                            verify(&result.unwrap_err(), N);
                        }
                        elapsed
                    });
                });
                group.finish();
            }
            input[N - 1] = 1;
            let mut output = [0_u32; N];
            super::source::pipeline(&input, &right, &factor)
                .unwrap()
                .read_into(&mut output)
                .unwrap();
            for (actual, input) in output.iter().zip(input) {
                assert_eq!(*actual, (input + 1) * 2);
            }
        }
    }
    global::clear_thread_cache().unwrap();
}

fn verify(error: &fusion_pcu::PcuExecutionError, n: usize) {
    let fault = error
        .arithmetic_fault()
        .expect("expected checked range fault");
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(fault.invocation_id, u64::try_from(n - 1).unwrap());
    assert!(!fault.recovered);
}

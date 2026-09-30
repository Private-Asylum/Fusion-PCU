//! Generated-call acceptance across both compiled hosted providers.
#![cfg(all(feature = "cuda", feature = "rocm"))]

use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::global::{
    self,
    PcuBackendChoice,
    PcuExecutionError,
    PcuExecutionPolicy,
};

static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[pcu(invocations = N)]
fn dual_provider_transform<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 3.0 + 1.0;
}

fn assert_output(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
#[ignore = "requires an RX 6900 XT ROCm device and a usable ROCm compiler; CUDA must be unavailable"]
#[allow(clippy::suboptimal_flops)] // Match the generated kernel's separate multiply/add rounding.
fn generated_calls_follow_automatic_and_explicit_rocm_policy_transitions() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();

    let input = [2.0_f32, -3.0, 4.5];
    let expected = input.map(|value| value * 3.0 + 1.0);
    let mut output = [f32::NAN; 3];

    // Automatic selection must find and execute on the available AMD device.
    global::use_defaults().unwrap();
    dual_provider_transform(&input, &mut output).unwrap();
    assert_output(&output, &expected);

    // A policy update must route the already generated call through explicit ROCm and retain
    // the caller's device ordinal and new cache policy.
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Rocm,
        device: Some(0),
        cache_capacity: 1,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    output.fill(f32::NAN);
    dual_provider_transform(&input, &mut output).unwrap();
    assert_output(&output, &expected);

    // An explicit invalid ordinal cannot silently substitute the valid device 0.
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Rocm,
        device: Some(u32::MAX),
        cache_capacity: 1,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    output.fill(91.0);
    assert!(dual_provider_transform(&input, &mut output).is_err());
    assert_output(&output, &[91.0; 3]);

    // The dual-feature build exposes CUDA, but the AMD host has no compatible CUDA device.
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Cuda,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    assert!(matches!(
        dual_provider_transform(&input, &mut output),
        Err(PcuExecutionError::NoBackendEnabled
            | PcuExecutionError::BackendFailure(_)
            | PcuExecutionError::NoCompatibleDevice(_))
    ));
    assert_output(&output, &[91.0; 3]);

    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Rocm,
        device: Some(0),
        cache_capacity: 1,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    output.fill(f32::NAN);
    dual_provider_transform(&input, &mut output).unwrap();
    assert_output(&output, &expected);

    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
}

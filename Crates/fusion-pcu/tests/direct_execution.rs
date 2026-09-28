//! Hardware conformance for direct calls through the public facade.
#![cfg(feature = "rocm")]

use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::global::{
    self,
    PcuBackendChoice,
    PcuExecutionPolicy,
};

static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn assert_f32_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[pcu(invocations = N)]
fn direct_transform<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 2.0 + 1.0;
}

#[pcu(invocations: N)]
fn direct_accumulate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = output[id] + input[id];
}

#[pcu(invocations: N)]
fn direct_factor<'input, 'factor, 'output, const N: usize>(
    input: &'input [f32; N],
    factor: &'factor f32,
    output: &'output mut [f32; N],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * factor;
}

#[pcu(invocations: N)]
fn direct_seed<'input, 'seed, 'output, const N: usize>(
    input: &'input [f64; N],
    seed: &'seed f64,
    output: &'output mut [f64; N],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + *seed;
}

#[test]
#[ignore = "requires a working ROCm device"]
fn readonly_scalar_parameters_use_current_values_in_cached_calls() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let input = [1.0_f32, 2.0, 4.0];
    let mut output = [0.0_f32; 3];
    for factor in [0.5_f32, 2.0, -1.0] {
        direct_factor(&input, &factor, &mut output).unwrap();
        assert_f32_bits(&output, &input.map(|value| value * factor));
    }
    // Generic storage signatures must retain concrete-reference coercions for scalar seeds.
    let mut boxed_factor = Box::new(0.25_f32);
    direct_factor(&input, &boxed_factor, &mut output).unwrap();
    assert_f32_bits(&output, &input.map(|value| value * 0.25));
    *boxed_factor = -2.0;
    direct_factor(&input, &boxed_factor, &mut output).unwrap();
    assert_f32_bits(&output, &input.map(|value| value * -2.0));
    let shared_factor = std::rc::Rc::new(4.0_f32);
    direct_factor(&input, &shared_factor, &mut output).unwrap();
    assert_f32_bits(&output, &input.map(|value| value * 4.0));
    let shared_factor = std::sync::Arc::new(-0.5_f32);
    direct_factor(&input, &shared_factor, &mut output).unwrap();
    assert_f32_bits(&output, &input.map(|value| value * -0.5));
    let input = [0.5_f64, 2.0, 4.0];
    let mut output = [0.0_f64; 3];
    for seed in [0.25_f64, -4.0, 2.0] {
        direct_seed(&input, &seed, &mut output).unwrap();
        assert_eq!(
            output.map(f64::to_bits),
            input.map(|value| (value + seed).to_bits())
        );
    }
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
#[allow(clippy::suboptimal_flops)] // Match the kernel's separate multiply/add rounding.
fn default_calls_specialize_reuse_and_preserve_fresh_host_values() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    for round in 0_u8..4 {
        let input = [f32::from(round), 2.0, 3.0, 4.0];
        let tail = -7.0 - f32::from(round);
        let mut output = [tail; 6];
        direct_transform::<4>(&input, &mut output).unwrap();
        assert_f32_bits(&output[..4], &input.map(|value| value * 2.0 + 1.0));
        assert_f32_bits(&output[4..], &[tail; 2]);
        // A separate read/modify/write entry must upload current RAM rather than reusing old
        // cached output contents. Its unprocessed tail still belongs to the caller.
        output.fill(tail);
        direct_accumulate::<4>(&input, &mut output).unwrap();
        assert_f32_bits(&output[..4], &input.map(|value| tail + value));
        assert_f32_bits(&output[4..], &[tail; 2]);
    }
    let mut output = [-7.0; 4];
    direct_transform::<2>(&[5.0, 6.0], &mut output).unwrap();
    assert_f32_bits(&output, &[11.0, 13.0, -7.0, -7.0]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn explicit_missing_device_never_substitutes_another_device() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Rocm,
        device: Some(u32::MAX),
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut output = [123.0; 2];
    assert!(direct_transform::<2>(&[1.0, 2.0], &mut output).is_err());
    assert_f32_bits(&output, &[123.0; 2]);
    global::use_defaults().unwrap();
    direct_transform::<2>(&[1.0, 2.0], &mut output).unwrap();
    assert_f32_bits(&output, &[3.0, 5.0]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
#[allow(clippy::suboptimal_flops)] // Match the kernel's separate multiply/add rounding.
fn separate_threads_own_their_executables() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    for value in [1.0, 2.0] {
        std::thread::spawn(move || {
            let mut output = [0.0; 2];
            direct_transform::<2>(&[value; 2], &mut output).unwrap();
            assert_f32_bits(&output, &[value * 2.0 + 1.0; 2]);
        })
        .join()
        .unwrap();
    }
}

#[pcu(invocations = N)]
fn direct_checked<const N: usize>(a: &[u32], b: &[u32], q: &mut [u32], r: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(a[id], b[id]);
    q[id] = quotient;
    r[id] = remainder;
}

#[test]
#[ignore = "requires a working ROCm device"]
fn checked_error_preserves_host_output_and_cached_call_recovers() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let mut quotient = [91; 2];
    let mut remainder = [92; 2];
    assert!(direct_checked::<2>(&[10, 11], &[2, 0], &mut quotient, &mut remainder).is_err());
    assert_eq!(quotient, [91; 2]);
    assert_eq!(remainder, [92; 2]);
    direct_checked::<2>(&[10, 11], &[2, 3], &mut quotient, &mut remainder).unwrap();
    assert_eq!(quotient, [5, 3]);
    assert_eq!(remainder, [0, 2]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
fn bounded_cache_eviction_and_policy_updates_rebuild_without_stale_results() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::configure(PcuExecutionPolicy {
        cache_capacity: 1,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut output = [0.0; 4];
    direct_transform::<4>(&[1.0; 4], &mut output).unwrap();
    assert_f32_bits(&output, &[3.0; 4]);
    direct_transform::<2>(&[2.0; 2], &mut output).unwrap();
    assert_f32_bits(&output, &[5.0, 5.0, 3.0, 3.0]);
    direct_transform::<4>(&[3.0; 4], &mut output).unwrap();
    assert_f32_bits(&output, &[7.0; 4]);
    global::use_defaults().unwrap();
    direct_transform::<4>(&[4.0; 4], &mut output).unwrap();
    assert_f32_bits(&output, &[9.0; 4]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
#[allow(clippy::unnecessary_mut_passed)] // Exercise readable mutable carriers, not just shared refs.
fn ordinary_host_reference_coercions_survive_source_carrier_generalization() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    let mut input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    direct_transform::<4>(&mut input, &mut output).unwrap();
    assert_f32_bits(&output, &[3.0, 5.0, 7.0, 9.0]);
    input[0] = 8.0;
    direct_transform::<4>(&mut input[..], &mut output[..]).unwrap();
    assert_f32_bits(&output, &[17.0, 5.0, 7.0, 9.0]);
    let mut input = vec![2.0_f32, 4.0, 6.0, 8.0];
    let mut output = vec![0.0_f32; 4];
    direct_transform::<4>(&input, &mut output).unwrap();
    assert_f32_bits(&output, &[5.0, 9.0, 13.0, 17.0]);
    input[1] = -4.0;
    direct_transform::<4>(&mut input, &mut output).unwrap();
    assert_f32_bits(&output, &[5.0, -7.0, 13.0, 17.0]);
    let mut input = [1.0_f32, 2.0, 3.0];
    let mut factor = 2.0_f32;
    let mut output = [0.0_f32; 3];
    direct_factor(&mut input, &mut factor, &mut output).unwrap();
    assert_f32_bits(&output, &[2.0, 4.0, 6.0]);
    factor = -1.0;
    direct_factor(&input, &factor, &mut output).unwrap();
    assert_f32_bits(&output, &[-1.0, -2.0, -3.0]);
    global::clear_thread_cache().unwrap();
}

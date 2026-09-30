#![cfg(feature = "cuda")]

#[cfg(not(feature = "rocm"))]
#[rustfmt::skip]
use fusion_pcu::global::{
    self,
    PcuBackendChoice,
    PcuExecutionError,
    PcuExecutionPolicy,
};
use fusion_pcu::pcu;

#[pcu(invocations = N)]
fn cuda_facade_transform<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let index = pcu::context::global_invocation_id();
    output[index] = input[index] * 2.0;
}

#[test]
#[cfg(not(feature = "rocm"))]
fn explicit_uncompiled_rocm_backend_is_rejected_without_cpu_fallback() {
    let policy = PcuExecutionPolicy {
        backend: PcuBackendChoice::Rocm,
        ..PcuExecutionPolicy::default()
    };
    global::configure(policy).unwrap();

    let input = [1.0_f32];
    let mut output = [0.0_f32];
    assert!(matches!(
        cuda_facade_transform(&input, &mut output),
        Err(PcuExecutionError::NoBackendEnabled)
    ));
    global::use_defaults().unwrap();
}

#[test]
#[ignore = "requires a CUDA device and a usable CUDA compiler"]
#[cfg(not(feature = "rocm"))]
fn cuda_only_feature_runs_generated_calls_with_automatic_and_explicit_cuda() {
    let input = [2.0_f32, -3.0];
    let mut output = [0.0_f32; 2];

    global::use_defaults().unwrap();
    cuda_facade_transform(&input, &mut output).unwrap();
    assert_eq!(output.map(f32::to_bits), [4.0_f32, -6.0].map(f32::to_bits));

    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Cuda,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    output = [0.0; 2];
    cuda_facade_transform(&input, &mut output).unwrap();
    assert_eq!(output.map(f32::to_bits), [4.0_f32, -6.0].map(f32::to_bits));
    global::use_defaults().unwrap();
}

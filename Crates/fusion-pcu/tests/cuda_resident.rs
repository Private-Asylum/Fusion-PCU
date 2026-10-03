//! Ordinary source signatures across CUDA owned, borrowed, mutable and host storage.
#![cfg(all(feature = "cuda", feature = "tensor"))]

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu]
fn identity(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn activate(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::relu(input)
}

#[pcu]
fn consumed_pair(
    lhs: PcuTensor<f32>,
    rhs: PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn consumed_mixed(lhs: PcuTensor<f32>, rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu]
fn choose(
    lhs: PcuTensor<f32>,
    _unused: PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(lhs)
}

#[pcu(flag(non_strict))]
fn checked_matmul(
    lhs: &[[f32; 2]; 2],
    rhs: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(lhs, rhs)
}

#[pcu(invocations: N)]
fn accumulate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = output[id] + input[id];
}

#[pcu]
fn checked_pair(lhs: &[u32], rhs: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    Ok(pcu::identity(lhs)? + rhs)
}

fn configure_cuda() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
}

#[test]
#[ignore = "requires a working CUDA device"]
fn ordinary_cuda_source_ownership_and_fault_contracts() {
    configure_cuda();
    let input = [-1.0_f32, 2.0, -3.0, 4.0];
    let resident_input = identity(&input).unwrap();
    let mut output = identity(&[10.0_f32; 4]).unwrap();
    accumulate::<4>(&resident_input, &mut output).unwrap();
    accumulate::<4>(&input, &mut output).unwrap();
    let mut actual = [0.0_f32; 4];
    output.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [8.0, 14.0, 4.0, 18.0].map(f32::to_bits)
    );

    let mut host_output = [20.0_f32; 4];
    accumulate::<4>(&resident_input, &mut host_output).unwrap();
    assert_eq!(
        host_output.map(f32::to_bits),
        [19.0, 22.0, 17.0, 24.0].map(f32::to_bits)
    );
    let result = activate(resident_input).unwrap();
    assert_eq!(result.shape(), &[4]);
    let peer = identity(&[1.0_f32; 4]).unwrap();
    let result = consumed_pair(result, peer).unwrap();
    let result = consumed_mixed(result, &[2.0_f32; 4]).unwrap();
    result.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [3.0, 5.0, 3.0, 7.0].map(f32::to_bits)
    );

    // The same source specialization captures exact extent metadata on a cold shape miss.
    let shorter = identity(&[5.0_f32, 6.0]).unwrap();
    assert_eq!(shorter.shape(), &[2]);
    global::clear_thread_cache().unwrap();
    result.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [3.0, 5.0, 3.0, 7.0].map(f32::to_bits)
    );
    accumulate::<4>(&result, &mut output).unwrap();
    output.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [11.0, 19.0, 7.0, 25.0].map(f32::to_bits)
    );

    let fault = checked_pair(&[u32::MAX], &[1]).unwrap_err();
    assert!(matches!(fault, PcuExecutionError::ArithmeticFault(_)));
    let recovered = checked_pair(&[2], &[3]).unwrap();
    let mut integer = [0_u32; 1];
    recovered.read_into(&mut integer).unwrap();
    assert_eq!(integer, [5]);

    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let conflict = accumulate::<4>(&result, &mut output).unwrap_err();
    assert!(matches!(
        conflict,
        PcuExecutionError::ResidentPolicyConflict
    ));
    configure_cuda();
    output.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [11.0, 19.0, 7.0, 25.0].map(f32::to_bits)
    );

    global::clear_thread_cache().unwrap();
    let isolated = identity(&[1.0_f32; 4]).unwrap();
    let conflict = accumulate::<4>(&isolated, &mut output).unwrap_err();
    assert!(matches!(
        conflict,
        PcuExecutionError::Argument(global::PcuArgumentError::SessionMismatch)
    ));
    output.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [11.0, 19.0, 7.0, 25.0].map(f32::to_bits)
    );
    // Pruned moved inputs do not impose their root affinity on the selected program.
    let retained = choose(output, isolated).unwrap();
    retained.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [11.0, 19.0, 7.0, 25.0].map(f32::to_bits)
    );

    let matrix = [[1.0_f32, 2.0], [3.0, 4.0]];
    let rejected = checked_matmul(&matrix, &matrix).unwrap_err();
    let PcuExecutionError::NoCompatibleDevice {
        rejected: rejections,
        ..
    } = rejected
    else {
        panic!("checked compound rejection retains cold candidate metadata: {rejected}");
    };
    assert!(rejections.iter().any(|(_, error)| matches!(
        error,
        PcuExecutionError::CudaTensorExecution(
            fusion_pcu_cuda::CudaTensorExecutionError::Unsupported { .. }
        )
    )));
    global::clear_thread_cache().unwrap();
}

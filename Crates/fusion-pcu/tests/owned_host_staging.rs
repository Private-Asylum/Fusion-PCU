//! Ordinary host borrows observe fresh inputs without exposing transfer machinery.
#![cfg(all(
    feature = "tensor",
    any(feature = "cpu", feature = "cuda", feature = "rocm")
))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuTensor,
};

#[pcu]
fn transform(
    left: &[u32],
    right: &[u32],
    factor: &[u32],
) -> Result<PcuTensor<u32>, PcuExecutionError> {
    let sum = pcu::add(left, right)?;
    pcu::mul(&sum, factor)
}

#[pcu]
fn projected(
    left: &[u32],
    right: &[u32],
    _unused: &[u32],
) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::add(left, right)
}

#[pcu]
fn retained(input: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn with_resident(
    input: &PcuTensor<u32>,
    right: &[u32],
) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::add(input, right)
}

fn verify(owner: &PcuTensor<u32>, expected: &[u32]) {
    let mut host = vec![0xa5a5_a5a5; expected.len() + 2];
    owner.read_into(&mut host).unwrap();
    assert_eq!(&host[..expected.len()], expected);
    assert_eq!(&host[expected.len()..], &[0xa5a5_a5a5; 2]);
}

fn shape(n: usize) {
    let mut left = vec![0_u32; n];
    let mut right = vec![1_u32; n];
    let mut factor = vec![2_u32; n];
    let original = transform(&left, &right, &factor).unwrap();
    let original_expected = vec![2_u32; n];
    for generation in 1_u32..=64 {
        for index in 0..n {
            left[index] = generation + u32::try_from(index % 7).unwrap();
            right[index] = generation + 1;
            factor[index] = generation % 3 + 1;
        }
        let expected: Vec<_> = left
            .iter()
            .zip(&right)
            .zip(&factor)
            .map(|((&left, &right), &factor)| (left + right) * factor)
            .collect();
        let owner = transform(&left, &right, &factor).unwrap();
        verify(&owner, &expected);
        // A formal argument can be absent from the selected physical binding set.
        let selected = projected(&left, &right, &factor).unwrap();
        let sum: Vec<_> = left.iter().zip(&right).map(|(a, b)| a + b).collect();
        verify(&selected, &sum);
        verify(&original, &original_expected);
    }
    left.fill(1);
    right.fill(1);
    factor.fill(2);
    // Original operation order beats a smaller lane in a later stage.
    left[7] = u32::MAX;
    factor[3] = u32::MAX;
    let fault = transform(&left, &right, &factor)
        .unwrap_err()
        .arithmetic_fault()
        .unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(fault.invocation_id, 7);
    assert!(!fault.recovered);
    verify(&original, &original_expected);
    left.fill(1);
    factor.fill(2);
    let retry = transform(&left, &right, &factor).unwrap();
    verify(&retry, &vec![4; n]);
    let resident = retained(&left).unwrap();
    let mixed = with_resident(&resident, &right).unwrap();
    global::clear_thread_cache().unwrap();
    verify(&mixed, &vec![2; n]);
    verify(&resident, &left);
    verify(&retry, &vec![4; n]);
    verify(&original, &original_expected);
    let recaptured = transform(&left, &right, &factor).unwrap();
    verify(&recaptured, &vec![4; n]);
}

fn run(backend: global::PcuBackendChoice) {
    for observation in [
        fusion_pcu::PcuExecutionObservationPolicy::Automatic,
        fusion_pcu::PcuExecutionObservationPolicy::HostObservedStages,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend,
            observation,
            ..Default::default()
        })
        .unwrap();
        for n in [65, 4096] {
            shape(n);
        }
    }
    global::clear_thread_cache().unwrap();
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_host_borrow_source_contract() {
    run(global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "Requires native CUDA; changing host-only and mixed resident source calls"]
fn cuda_host_borrow_source_contract() {
    run(global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "Requires native ROCm; changing all-host and mixed resident source calls"]
fn rocm_host_borrow_source_contract() {
    run(global::PcuBackendChoice::Rocm);
}

fn source_retention_survives_fifo(backend: global::PcuBackendChoice) {
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend,
        cache_capacity: 64,
        ..Default::default()
    })
    .unwrap();
    // Keep every source specialization alive in the facade cache while admitting 68 distinct
    // fixed Add/Mul keys into its shared 32-entry backend FIFO. Shape 65 is evicted there.
    for n in 65..99 {
        let owner = transform(&vec![1; n], &vec![2; n], &vec![3; n]).unwrap();
        verify(&owner, &vec![9; n]);
    }
    for generation in 1_u32..=64 {
        let owner = transform(&[generation; 65], &[2; 65], &[3; 65]).unwrap();
        verify(&owner, &[(generation + 2) * 3; 65]);
    }
    let mut overflowing = [1_u32; 65];
    overflowing[7] = u32::MAX;
    let fault = transform(&overflowing, &[2; 65], &[3; 65])
        .unwrap_err()
        .arithmetic_fault()
        .unwrap();
    assert_eq!(fault.invocation_id, 7);
    let retry = transform(&[4; 65], &[2; 65], &[3; 65]).unwrap();
    verify(&retry, &[18; 65]);
    global::clear_thread_cache().unwrap();
    verify(&retry, &[18; 65]);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "Requires native ROCm; genuine source retains fixed kernels across backend FIFO eviction"]
fn rocm_source_retention_survives_backend_fifo_eviction() {
    source_retention_survives_fifo(global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "Requires native CUDA; genuine source retains fixed kernels across backend FIFO eviction"]
fn cuda_source_retention_survives_backend_fifo_eviction() {
    source_retention_survives_fifo(global::PcuBackendChoice::Cuda);
}

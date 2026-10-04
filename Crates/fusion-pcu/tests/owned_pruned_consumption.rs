//! A consumed but unused owner must not select the execution session.
#![cfg(all(
    feature = "tensor",
    any(feature = "cpu", feature = "rocm", feature = "cuda")
))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu]
fn retain(input: &[u32; 3]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn pruned(_unused: PcuTensor<u32>) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { [7_u32; 3] })
}

#[pcu]
fn selected(input: PcuTensor<u32>) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::identity(input)
}

fn configure(backend: global::PcuBackendChoice, block_size: u32) {
    global::configure(global::PcuExecutionPolicy {
        backend,
        block_size,
        ..Default::default()
    })
    .unwrap();
}

fn run(backend: global::PcuBackendChoice) {
    configure(backend, 256);
    let older = retain(&[11, 13, 17]).unwrap();
    // Changing the block policy produces a distinct session; the unused donor's
    // older session must not constrain the graph's selected zero-input producer.
    let donors = (0..16)
        .map(|generation| retain(&[19 + generation, 23, 29]).unwrap())
        .collect::<Vec<_>>();
    configure(backend, 128);
    let mut stack = [31; 5];
    let mut outputs = Vec::new();
    for donor in donors {
        let output = pruned(donor).unwrap();
        output.read_into(&mut stack).unwrap();
        assert_eq!(stack, [7, 7, 7, 31, 31]);
        outputs.push(output);
    }
    global::clear_thread_cache().unwrap();
    for output in &outputs {
        output.read_into(&mut stack).unwrap();
        assert_eq!(stack, [7, 7, 7, 31, 31]);
    }
    older.read_into(&mut stack).unwrap();
    assert_eq!(stack, [11, 13, 17, 31, 31]);
    // A genuinely used donor still participates in session affinity and ownership.
    configure(backend, 256);
    let used = selected(older).unwrap();
    used.read_into(&mut stack).unwrap();
    assert_eq!(stack, [11, 13, 17, 31, 31]);
}

#[cfg(feature = "cpu")]
#[test]
fn sole_pruned_owner_uses_the_current_policy_and_retains_old_results() {
    run(global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "Requires native ROCm; CPU proof does not cover GPU consuming dispatch"]
fn rocm_pruned_consuming_source_contract() {
    run(global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "Requires native CUDA; CPU proof does not cover GPU consuming dispatch"]
fn cuda_pruned_consuming_source_contract() {
    run(global::PcuBackendChoice::Cuda);
}

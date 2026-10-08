//! One PCU scope versus deliberate intermediate observations in ordinary Rust.
//!
//! Run with `cargo run -p fusion-pcu --features rocm,tensor --example execution-boundaries`.
//! CUDA can also supply the runtime-selected provider.

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
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
fn add_stage(left: &[u32], right: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::add(left, right)
}

#[pcu]
fn multiply_stage(
    input: &PcuTensor<u32>,
    factor: &[u32],
) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::mul(input, factor)
}

fn main() -> Result<(), PcuExecutionError> {
    // Discovery and selection are lazy defaults. To request host observation between internal
    // checked stages, set observation: PcuExecutionObservationPolicy::HostObservedStages in
    // global::configure(...). This changes scheduling, not arithmetic strictness or residency.
    let left = [1_u32; 65];
    let right = [2_u32; 65];
    let factor = [3_u32; 65];
    let mut output = [0_u32; 65];
    {
        // One synchronous result boundary. RAM inputs are staged automatically. An eligible
        // provider may guard internal successors on-device and observe completion once. A
        // failed Add prevents Mul's user work; no partial result owner is returned on error.
        let combined = transform(&left, &right, &factor)?;
        // The valid result stays device-resident until this explicit read into stack RAM.
        combined.read_into(&mut output)?;
        assert_eq!(output, [9; 65]);

        // Deliberately split the same computation. This ordinary Rust call resolves Add's
        // promised errors before returning Ok; it does not download its payload automatically.
        let sum = add_stage(&left, &right)?;
        // Host code can now inspect an intermediate, log it, or decide whether to continue.
        let mut intermediate = [0_u32; 65];
        sum.read_into(&mut intermediate)?;
        println!("Intermediate in stack RAM: {intermediate:?}");
        // The read did not consume sum. Its original device bytes bind directly here; factor
        // moves from RAM. No GPU -> RAM -> GPU round trip is required for sum itself.
        let split = multiply_stage(&sum, &factor)?;
        split.read_into(&mut output)?;
        assert_eq!(output, [9; 65]);
        // Ordinary scope exit releases the result owners. Private prepared caches have their
        // own lifetime; dropping an owner does not promise an immediate driver allocation free.
    }
    println!("Result in stack RAM: {output:?}");
    global::clear_thread_cache()?;
    Ok(())
}

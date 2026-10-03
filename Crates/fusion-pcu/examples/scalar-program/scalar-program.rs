//! Ordered scalar statements with ordinary Rust borrows and lifted errors.
//!
//! ```toml
//! [dependencies]
//! fusion-pcu = { version = "0.0.7", features = ["cpu"] }
//! ```
//! Run: cargo run -p fusion-pcu --features cpu --example scalar-program
//!
//! This slice explicitly selects its qualified CPU implementation. Other
//! providers must separately admit ordered mutable working state and publication.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuExecutionFaultKind,
};

#[pcu(invocations: N)]
fn transform<T: PcuCheckedFloat, const N: usize>(
    input: &[T; N],
    intermediate: &mut [T; N],
    output: &mut [T; N],
) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    intermediate[id] = original + original;
    let mut updated = intermediate[id];
    updated = updated * original;
    output[id] = updated;
}

fn main() -> Result<(), PcuExecutionError> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut intermediate = [0.0; 4];
    let mut output = [0.0; 4];

    // The first call lowers and prepares once. Each scalar local is a typed
    // value, not a fresh allocation. Reassignment creates a new SSA value;
    // `original` still refers to its old value. Loads see earlier stores in order.
    // All arrays stay in caller-owned stack RAM on CPU; PCU uses private retained
    // working storage until both outputs can be published after completion.
    transform(&input, &mut intermediate, &mut output)?;
    println!("Intermediate in stack RAM: {intermediate:?}");
    println!("Output in stack RAM: {output:?}");

    // Warm calls reuse preparation while consuming fresh borrowed values.
    transform(&[2.0, 3.0, 4.0, 5.0], &mut intermediate, &mut output)?;
    let before = (intermediate, output);
    let error = transform(
        &[1.0, f32::INFINITY, 3.0, 4.0],
        &mut intermediate,
        &mut output,
    )
    .expect_err("the deliberately invalid floating operand must error");
    assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand
        && fault.invocation_id == 1 && !fault.recovered));
    assert_eq!((intermediate, output), before); // No partial result escaped.
    println!("Rejected input; previous complete outputs retained: {output:?}");

    // No host borrow survives the synchronous call. Clearing the cached plan
    // releases its private CPU working banks; the caller's stack arrays remain.
    global::clear_thread_cache()?;
    Ok(())
}

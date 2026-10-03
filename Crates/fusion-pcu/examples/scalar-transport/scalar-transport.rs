//! Representation-preserving work through an ordinary mutable Rust borrow.
//!
//! ```toml
//! [dependencies]
//! fusion-pcu = { version = "0.0.7", features = ["cpu"] }
//! ```
//! Run: cargo run -p fusion-pcu --features cpu --example scalar-transport
//!
//! This executable explicitly selects the CPU transport implementation. GPU
//! transport implementations require their own native qualification; a scalar
//! carrier's existence never promises arithmetic support on every device.

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuU512,
};

#[pcu(invocations: N)]
fn replace<T: PcuScalar, const N: usize>(
    values: &mut [T; N],
    replacement: &T,
    previous: &mut [T; N],
) {
    let id = pcu::context::global_invocation_id();
    let saved = values[id];
    values[id] = *replacement;
    previous[id] = saved;
}

fn main() -> Result<(), PcuExecutionError> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })?;

    let mut values = [-0.0_f64, f64::from_bits(0x7ff8_0000_0000_0042), 1.0];
    let original = values.map(f64::to_bits);
    let mut previous = [0.0; 3];
    // Preparation retains private working storage. The loaded local owns its
    // bits even after `values[id]` is overwritten. This is transport, so NaN
    // payloads and negative zero are copied without arithmetic normalization.
    // The synchronous call publishes both completed prefixes into stack RAM;
    // no host borrow or incomplete execution escapes its returned Result.
    replace(&mut values, &2.0, &mut previous)?;
    assert_eq!(previous.map(f64::to_bits), original);
    println!("Previous encodings in stack RAM: {original:016x?}");
    println!("Replacement values in stack RAM: {values:?}");

    let wide = PcuU512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, u64::MAX]);
    let mut values = [wide; 2];
    let mut previous = [PcuU512::ZERO; 2];
    replace(&mut values, &PcuU512::ZERO, &mut previous)?;
    assert_eq!(previous, [wide; 2]);
    assert_eq!(values, [PcuU512::ZERO; 2]);
    println!("Previous 512-bit values in stack RAM: {previous:?}");

    // CPU uses RAM throughout. Clearing retained plans releases their private
    // working banks; the returned stack values remain ordinary owned Rust data.
    global::clear_thread_cache()?;
    Ok(())
}

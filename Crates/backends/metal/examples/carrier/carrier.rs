//! Wide scalar bits remain exact through ordinary staged Metal broadcast and dense copy.
#[rustfmt::skip]
use pcu_facade::{global,pcu,PcuScalar,PcuU512};
#[pcu(invocations=65,flag(strict),crate_path=::pcu_facade)]
fn broadcast<T: PcuScalar>(seed: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = *seed;
}
#[pcu(invocations=65,crate_path=::pcu_facade)]
fn identity<T: PcuScalar>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        ..Default::default()
    })?;
    let seed = PcuU512::from_limbs_le([u64::MAX, 1, 2, 3, 4, 5, 6, 1 << 63]);
    let sentinel = PcuU512::from_limbs_le([91; 8]);
    let mut first = [sentinel; 68];
    let mut copied = [sentinel; 68];
    broadcast(&seed, &mut first)?;
    identity(&first, &mut copied)?;
    assert_eq!(copied[..65], [seed; 65]);
    assert_eq!(copied[65..], [sentinel; 3]);
    global::clear_thread_cache()?;
    global::use_defaults()?;
    println!(
        "Metal exact512-bit transport:65 values; high limb {:016x}; tails preserved",
        copied[0].to_limbs_le()[7]
    );
    Ok(())
}

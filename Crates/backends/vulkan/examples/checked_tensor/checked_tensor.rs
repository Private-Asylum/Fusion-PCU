//! Ordinary owned source keeps real native storage across a pointwise chain.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
    PcuI512,
};

#[pcu(crate_path=::pcu_facade)]
fn retained(input: &[PcuI512]) -> Result<PcuTensor<PcuI512>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
fn square_sum(
    left: &[PcuI512],
    right: &[PcuI512],
) -> Result<PcuTensor<PcuI512>, PcuExecutionError> {
    let sum = pcu::add(left, right)?;
    pcu::mul(&sum, &sum)
}
fn main() -> Result<(), PcuExecutionError> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })?;
    let one = PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]);
    let two = PcuI512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]);
    let nine = PcuI512::from_limbs_le([9, 0, 0, 0, 0, 0, 0, 0]);
    let a = retained(&[one; 65])?;
    let b = retained(&[two; 65])?;
    let result = square_sum(&a, &b)?;
    global::clear_thread_cache()?;
    drop(a);
    drop(b);
    let mut observed = [one; 68];
    result.read_into(&mut observed)?;
    assert_eq!(observed[..65], [nine; 65]);
    assert_eq!(observed[65..], [one; 3]);
    println!(
        "Vulkan ordinary I512 borrowed-owner Add/Mul: exact output survives cache clear and input drop."
    );
    global::use_defaults()?;
    Ok(())
}

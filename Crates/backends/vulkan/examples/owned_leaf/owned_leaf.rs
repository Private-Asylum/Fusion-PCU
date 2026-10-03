//! An ordinary source owner detaches wide raw bits and survives cache eviction/borrowed reuse.
#[rustfmt::skip]
use pcu_facade::{global,pcu,PcuF256Bits,PcuExecutionError,PcuScalar,PcuTensor};
#[pcu(crate_path=::pcu_facade)]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
fn main() -> Result<(), PcuExecutionError> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })?;
    let mut input = [PcuF256Bits::decode_le([0xff; 32]); 7];
    let first = retain(&input)?;
    input.fill(PcuF256Bits::decode_le([0; 32]));
    global::clear_thread_cache()?;
    let second = retain(&first)?;
    drop(first);
    let mut result = [PcuF256Bits::decode_le([0x73; 32]); 9];
    second.read_into(&mut result)?;
    assert!(
        result[..7]
            .iter()
            .all(|value| value.encode_le() == [0xff; 32])
    );
    assert!(
        result[7..]
            .iter()
            .all(|value| value.encode_le() == [0x73; 32])
    );
    println!(
        "Vulkan owned F256 carrier: exact raw bits, detached host, cache-clear lifetime, borrowed transfer and preserved readback tail"
    );
    global::use_defaults()?;
    Ok(())
}

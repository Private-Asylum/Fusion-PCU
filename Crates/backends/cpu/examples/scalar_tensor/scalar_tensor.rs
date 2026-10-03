//! Wide floating representations remain transportable without claiming arithmetic.
#[rustfmt::skip]
use pcu_facade::{pcu,global,PcuScalar,PcuTensor,PcuExecutionError,PcuF256Bits};
#[pcu(crate_path=::pcu_facade)]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let input = [
        PcuF256Bits::decode_le([255; 32]),
        PcuF256Bits::decode_le(std::array::from_fn(|i| u8::try_from(i).unwrap())),
    ];
    let owner = retain(&input).unwrap();
    let sibling = retain(&owner).unwrap();
    drop(owner);
    global::clear_thread_cache().unwrap();
    let sentinel = PcuF256Bits::decode_le([17; 32]);
    let mut observed = [sentinel; 4];
    sibling.read_into(&mut observed).unwrap();
    assert_eq!(observed[0].encode_le(), input[0].encode_le());
    assert_eq!(observed[1].encode_le(), input[1].encode_le());
    assert_eq!(observed[2..], [sentinel; 2]);
    println!(
        "CPU F256 raw tensor owners preserve complete bytes, sibling lifetime and caller tails"
    );
    global::use_defaults().unwrap();
}

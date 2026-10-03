//! CPU ordinary low-format tensor owners, useful exact arithmetic and private fatal lifetime.
#[rustfmt::skip]
use pcu_facade::{pcu,global,PcuCheckedFloat,PcuTensor,PcuExecutionError,PcuF8E4M3FnBits};
#[pcu(crate_path=::pcu_facade)]
fn add<T: PcuCheckedFloat>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}
#[pcu(crate_path=::pcu_facade)]
fn relu<T: PcuCheckedFloat>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let input = [
        PcuF8E4M3FnBits::from_bits(0x38),
        PcuF8E4M3FnBits::from_bits(0xb8),
    ];
    let output = add(&input, &input).unwrap();
    let sibling = relu(&output).unwrap();
    let mut observed = [PcuF8E4M3FnBits::from_bits(0x17); 4];
    sibling.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(PcuF8E4M3FnBits::to_bits),
        [0x40, 0, 0x17, 0x17]
    );
    assert!(add(&[PcuF8E4M3FnBits::from_bits(0x7f)], &input[..1]).is_err());
    output.read_into(&mut observed).unwrap();
    assert_eq!(observed[0].to_bits(), 0x40);
    println!("ordinary CPU FP8 tensor arithmetic retains independent owners and caller tails");
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

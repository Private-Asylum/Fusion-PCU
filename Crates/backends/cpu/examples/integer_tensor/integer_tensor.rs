//! Full-width ordinary owned source keeps sibling owners readable through checked failure.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuScalar,PcuTensor,PcuExecutionError,PcuU512,global};
#[pcu(crate_path=::pcu_facade)]
fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
fn add<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let one = PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]);
    let mut limbs = [0; 8];
    limbs[7] = 1;
    let input = [PcuU512::from_limbs_le(limbs); 7];
    let owner = identity(&input).unwrap();
    let sibling = identity(&owner).unwrap();
    let result = add(&owner, &[one; 7]).unwrap();
    let mut read = [PcuU512::ZERO; 10];
    result.read_into(&mut read).unwrap();
    assert_eq!(read[0].to_limbs_le()[7], 1);
    assert_eq!(read[0].to_limbs_le()[0], 1);
    assert!(add(&owner, &[PcuU512::MAX; 7]).is_err());
    sibling.read_into(&mut read).unwrap();
    assert_eq!(&read[..7], &input);
    assert_eq!(&read[7..], &[PcuU512::ZERO; 3]);
    add(&owner, &[one; 7]).unwrap();
    println!(
        "U512 tensor: high limbs, escaping owners, sibling preservation, tails and retry pass"
    );
}

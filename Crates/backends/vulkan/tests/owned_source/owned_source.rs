//! Ordinary source uses real Vulkan backing; policy changes never migrate borrowed owners.
#[path = "../scalar_transport/sample/sample.rs"]
mod sample;
#[path = "../../../cpu/tests/scalar_tensor/source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{global,pcu,PcuScalar,PcuF256Bits,PcuExecutionError,PcuTensor,PcuRangePolicy};
use sample::{Sample, same};
#[pcu(crate_path=::pcu_facade)]
fn matrix<T: PcuScalar>(input: &[[T; 3]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
fn arithmetic(
    left: &[PcuF256Bits],
    right: &[PcuF256Bits],
) -> Result<PcuTensor<PcuF256Bits>, PcuExecutionError> {
    pcu::add(left, right)
}
#[test]
#[ignore = "requires an actual Vulkan GPU and exclusive correctness window"]
fn detached_owner_cache_session_policy_and_retry() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let banks: [Vec<PcuF256Bits>; 2] = std::array::from_fn(|bank| {
        (0..65)
            .map(|index| PcuF256Bits::pattern(index + bank * 97))
            .collect()
    });
    let first = source::identity(&banks[0]).unwrap();
    let second = source::identity(&banks[1]).unwrap();
    global::clear_thread_cache().unwrap();
    let cloned = source::identity(&first).unwrap();
    drop(first);
    let mut output = vec![PcuF256Bits::pattern(11); 68];
    cloned.read_into(&mut output).unwrap();
    same(&output[..65], &banks[0]);
    let consumed = source::consume(second).unwrap();
    consumed.read_into(&mut output).unwrap();
    same(&output[..65], &banks[1]);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        device: Some(1),
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        source::identity(&cloned),
        Err(PcuExecutionError::ResidentPolicyConflict)
    ));
    cloned.read_into(&mut output).unwrap();
    same(&output[..65], &banks[0]);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Automatic,
        ..Default::default()
    })
    .unwrap();
    let automatic = source::identity(&cloned).unwrap();
    drop(cloned);
    automatic.read_into(&mut output).unwrap();
    same(&output[..65], &banks[0]);
    for tail in &output[65..] {
        assert_eq!(tail.encode_le(), PcuF256Bits::pattern(11).encode_le());
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[test]
#[ignore = "requires an actual Vulkan GPU and exclusive correctness window"]
fn exact_leaf_shapes_empty_transport_and_no_wide_float_arithmetic_or_clamp() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let empty = source::identity::<PcuF256Bits>(&[]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.shape(), &[0]);
    let mut tail = [PcuF256Bits::pattern(17); 3];
    empty.read_into(&mut tail).unwrap();
    same(&tail, &[PcuF256Bits::pattern(17); 3]);
    let input: [[PcuF256Bits; 3]; 2] = std::array::from_fn(|row| {
        std::array::from_fn(|col| PcuF256Bits::pattern(row * 3 + col + 23))
    });
    let first = matrix(&input).unwrap();
    assert_eq!(first.shape(), &[2, 3]);
    global::clear_thread_cache().unwrap();
    let second = matrix(&first).unwrap();
    drop(first);
    let mut output = [PcuF256Bits::pattern(17); 8];
    second.read_into(&mut output).unwrap();
    same(&output[..6], input.as_flattened());
    same(&output[6..], &[PcuF256Bits::pattern(17); 2]);
    assert!(arithmetic(&[PcuF256Bits::pattern(1)], &[PcuF256Bits::pattern(2)]).is_err());
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        range_policy: PcuRangePolicy::Clamp,
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        matrix(&second),
        Err(PcuExecutionError::UnsupportedRangePolicy)
    ));
    second.read_into(&mut output).unwrap();
    same(&output[..6], input.as_flattened());
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let retry = matrix(&second).unwrap();
    drop(second);
    retry.read_into(&mut output).unwrap();
    same(&output[..6], input.as_flattened());
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

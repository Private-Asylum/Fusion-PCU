//! Ordinary Automatic calls must reach the sole compiled Vulkan provider.
#![cfg(all(
    feature = "vulkan",
    feature = "tensor",
    not(any(
        feature = "cpu",
        feature = "metal",
        feature = "mlx",
        feature = "rocm",
        feature = "cuda"
    ))
))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu]
fn initial<const N: usize>() -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { [7_u32; N] })
}

#[pcu]
fn shift<const N: usize>(input: &[u32; N]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    let bias = pcu::constant(const { [7_u32; N] })?;
    pcu::add(input, &bias)
}

#[pcu]
fn matrix() -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { [[1_u32, 2, 3], [4, 5, 6]] })
}

#[test]
#[ignore = "Requires a native Vulkan device; build with only vulkan,tensor provider features."]
fn automatic_default_executes_host_no_argument_matrix_and_retained_owner_calls() {
    global::configure(global::PcuExecutionPolicy::default()).unwrap();
    let first = initial::<65>().unwrap();
    let input = core::array::from_fn::<_, 65, _>(|i| u32::try_from(i).unwrap());
    let shifted = shift(&input).unwrap();
    let replay = initial::<65>().unwrap();
    let shaped = matrix().unwrap();
    assert_eq!(first.shape(), [65]);
    assert_eq!(shaped.shape(), [2, 3]);
    global::clear_thread_cache().unwrap();
    let mut host = [0xfeed_face_u32; 67];
    for owner in [&first, &replay] {
        owner.read_into(&mut host).unwrap();
        assert_eq!(&host[..65], &[7_u32; 65]);
        assert_eq!(&host[65..], &[0xfeed_face; 2]);
    }
    shifted.read_into(&mut host).unwrap();
    assert_eq!(host[..65], input.map(|v| v + 7));
    assert_eq!(&host[65..], &[0xfeed_face; 2]);
    let mut row_major = [0xfeed_face_u32; 8];
    shaped.read_into(&mut row_major).unwrap();
    assert_eq!(row_major, [1, 2, 3, 4, 5, 6, 0xfeed_face, 0xfeed_face]);
    global::clear_thread_cache().unwrap();
}

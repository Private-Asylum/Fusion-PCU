//! Rank is literal data, rather than an inference from its flattened byte count.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuFloatUnderflowPolicy,
    PcuTensor,
    PcuU512,
    PcuScalar,
};

#[pcu]
fn matrix<const R: usize, const C: usize, const VALUE: usize>()
-> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { [[if VALUE == 7 { 7_u32 } else { 11_u32 }; C]; R] })
}

#[pcu]
fn wide() -> Result<PcuTensor<PcuU512>, PcuExecutionError> {
    pcu::constant(
        const {
            [
                [
                    PcuU512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, u64::MAX]),
                    PcuU512::ZERO,
                ],
                [
                    PcuU512::ZERO,
                    PcuU512::from_limbs_le([8, 9, 10, 11, 12, 13, 14, 15]),
                ],
            ]
        },
    )
}

#[pcu]
fn consume(input: PcuTensor<u32>) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(invocations = 6)]
pub(super) fn overwrite<T: PcuScalar>(input: &[[T; 3]; 2], output: &mut [[T; 3]; 2]) {
    let id = pcu::context::global_invocation_id();
    let row = id / 3;
    let column = id % 3;
    output[row][column] = input[row][column];
}

fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}

#[test]
fn genuine_matrix_literals_keep_rank_cache_identity_and_escaped_ownership() {
    configure();
    let captured = global::__pcu_capture_tensor_program::<u32, 0, _>(
        [],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Strict,
        PcuNumericalOptions::default(),
        matrix::__pcu_capture_entry::<2, 3, 7>,
    )
    .unwrap();
    assert!(captured.input_values().is_empty());
    assert!(captured.argument_indices().is_empty());
    let mut owner = matrix::<2, 3, 7>().unwrap();
    assert_eq!(owner.shape(), [2, 3]);
    let same_size = matrix::<3, 2, 11>().unwrap();
    assert_eq!(same_size.shape(), [3, 2]);
    let mut stack = [99; 8];
    same_size.read_into(&mut stack).unwrap();
    assert_eq!(stack, [11, 11, 11, 11, 11, 11, 99, 99]);
    overwrite(&[[1, 2, 3], [4, 5, 6]], &mut owner).unwrap();
    let consumed = consume(owner).unwrap();
    let replay = matrix::<2, 3, 7>().unwrap();
    assert_eq!(replay.shape(), [2, 3]);
    replay.read_into(&mut stack).unwrap();
    assert_eq!(stack, [7, 7, 7, 7, 7, 7, 99, 99]);
    global::clear_thread_cache().unwrap();
    consumed.read_into(&mut stack).unwrap();
    assert_eq!(stack, [1, 2, 3, 4, 5, 6, 99, 99]);
    same_size.read_into(&mut stack).unwrap();
    assert_eq!(stack, [11, 11, 11, 11, 11, 11, 99, 99]);
    let wide = wide().unwrap();
    assert_eq!(wide.shape(), [2, 2]);
    let mut limbs = [PcuU512::ZERO; 5];
    wide.read_into(&mut limbs).unwrap();
    assert_eq!(limbs[0].to_limbs_le(), [1, 2, 3, 4, 5, 6, 7, u64::MAX]);
    assert_eq!(limbs[1], PcuU512::ZERO);
    assert_eq!(limbs[2], PcuU512::ZERO);
    assert_eq!(limbs[3].to_limbs_le(), [8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(limbs[4], PcuU512::ZERO);
}

#[test]
fn empty_matrix_constants_preserve_both_zero_extent_shapes_and_destination_tails() {
    configure();
    for owner in [matrix::<0, 3, 7>().unwrap(), matrix::<2, 0, 7>().unwrap()] {
        assert!(owner.is_empty());
        let mut tail = [19_u32; 3];
        owner.read_into(&mut tail).unwrap();
        assert_eq!(tail, [19; 3]);
    }
    assert_eq!(matrix::<0, 3, 7>().unwrap().shape(), [0, 3]);
    assert_eq!(matrix::<2, 0, 7>().unwrap().shape(), [2, 0]);
}

use fusion_pcu_macros::pcu;
use pcu_alias::{PcuDispatchDataOp, PcuDispatchOp, PcuScalar};

extern crate pcu_alias;

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn matrix_copy<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    let row = id / C;
    let col = id % C;
    output[row][col] = input[row][col];
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn matrix_copy_grid<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < R * C {
        let row = id / C;
        let col = id % C;
        output[row][col] = input[row][col];
        id += stride;
    }
}

#[test]
fn immutable_matrix_coordinates_lower_to_flat_invocation_indices() {
    let bindings = matrix_copy_bindings::<i64>();
    let kernel = matrix_copy_ir::<i64, 2, 5>(&bindings).expect("rank-two locals lower");
    assert_eq!(kernel.ir().entry.logical_shape, [10, 1, 1]);
    assert!(matches!(
        kernel.ir().ops,
        [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                index: pcu_alias::PcuDispatchIndex::InvocationId,
                ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                index: pcu_alias::PcuDispatchIndex::InvocationId,
                ..
            }),
            PcuDispatchOp::Control(pcu_alias::PcuDispatchControlOp::Return),
        ]
    ));
}

#[test]
fn rank_two_grid_stride_coordinates_cover_a_larger_shape_than_launch() {
    let bindings = matrix_copy_grid_bindings::<u64>();
    let kernel =
        matrix_copy_grid_ir::<u64, 2, 5>(&bindings).expect("rank-two grid-stride locals lower");
    assert_eq!(kernel.ir().entry.logical_shape, [3, 1, 1]);
    let [
        PcuDispatchOp::GridStrideLoop { extent, body },
        PcuDispatchOp::Control(pcu_alias::PcuDispatchControlOp::Return),
    ] = kernel.ir().ops
    else {
        panic!("rank-two grid-stride identity is one loop region and return")
    };
    assert_eq!(*extent, 10);
    assert!(matches!(
        body,
        [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                index: pcu_alias::PcuDispatchIndex::GridStrideId,
                ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                index: pcu_alias::PcuDispatchIndex::GridStrideId,
                ..
            }),
        ]
    ));
}

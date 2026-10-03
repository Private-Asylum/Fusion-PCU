//! All-carrier transport source is typed and ordered without numeric permission.

use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    describe_scalar_transport_map,
    validate_checked_float_map_kernel,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchOp,
    PcuScalar,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuValueType,
    PcuValueTypeCaps,
};

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn ordered<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    stage[id] = original;
    let mut selected = stage[id];
    selected = *seed;
    output[id] = selected;
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn grid<T: PcuScalar, const N: usize>(input: &[T], stage: &mut [T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let value = input[id];
        stage[id] = value;
        let updated = stage[id];
        output[id] = updated;
        id += stride;
    }
}

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn matrix<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    stage: &mut [[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    let row = id / C;
    let col = id % C;
    let original = input[row][col];
    stage[row][col] = original;
    let updated = stage[row][col];
    output[row][col] = updated;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn simple<T: PcuScalar, const N: usize>(
    ghost: &mut [T],
    output: &mut [T],
    unused: &[T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

fn assert_simple_source<T: PcuScalar>() {
    let bindings = simple_bindings::<T>();
    simple_ir::<T, 7>(&bindings).unwrap().with_ir(|ir| {
        let description = describe_scalar_transport_map::<2>(ir, T::TYPE).unwrap();
        assert_eq!(ir.bindings.len(), 4);
        assert_eq!(ir.bindings[0].access, PcuBindingAccess::ReadWrite);
        assert_eq!(ir.bindings[2].access, PcuBindingAccess::ReadOnly);
        assert_eq!(description.resources().len(), 2);
        assert!(description.resource(PcuBindingRef::new(0, 0)).is_none());
        assert!(description.resource(PcuBindingRef::new(0, 2)).is_none());
        let source = description.resource(PcuBindingRef::new(0, 3)).unwrap();
        let destination = description.resource(PcuBindingRef::new(0, 1)).unwrap();
        assert_eq!(source.minimum_initial_read_elements, 7);
        assert_eq!(source.minimum_write_elements, 0);
        assert_eq!(destination.minimum_read_elements, 0);
        assert_eq!(destination.minimum_write_elements, 7);
        assert_eq!(destination.declared_access, PcuBindingAccess::ReadWrite);
        assert_eq!(description.resources()[0].binding, source.binding);
        assert_eq!(description.resources()[1].binding, destination.binding);
        assert_eq!(
            ir.fully_written_binding_elements(destination.binding, 7),
            Some(7)
        );
    });
}

fn assert_source<T: PcuScalar>() {
    assert_simple_source::<T>();
    let bindings = ordered_bindings::<T>();
    ordered_ir::<T, 7>(&bindings).unwrap().with_ir(|ir| {
        let description = describe_scalar_transport_map::<4>(ir, T::TYPE).unwrap();
        assert_eq!(description.logical_extent, 7);
        assert_eq!(description.resources().len(), 4);
        assert_eq!(
            description
                .resource(PcuBindingRef::new(0, 0))
                .unwrap()
                .minimum_initial_read_elements,
            7
        );
        assert_eq!(
            description
                .resource(PcuBindingRef::new(0, 1))
                .unwrap()
                .minimum_initial_read_elements,
            1
        );
        assert!(description.resource(PcuBindingRef::new(0, 2)).is_none());
        assert_eq!(ir.bindings[2].access, PcuBindingAccess::ReadWrite);
        assert_eq!(
            description
                .resource(PcuBindingRef::new(0, 3))
                .unwrap()
                .minimum_initial_read_elements,
            0
        );
        assert_eq!(
            ir.fully_written_binding_elements(PcuBindingRef::new(0, 4), 7),
            Some(7)
        );
        assert!(
            validate_checked_float_map_kernel(
                ir,
                PcuValueType::Scalar(T::TYPE),
                PcuValueTypeCaps::for_scalar(T::TYPE)
            )
            .is_err()
        );
    });
    let bindings = grid_bindings::<T>();
    grid_ir::<T, 19>(&bindings).unwrap().with_ir(|ir| {
        let description = describe_scalar_transport_map::<3>(ir, T::TYPE).unwrap();
        assert_eq!(description.logical_extent, 19);
        assert_eq!(description.submitted_invocations, 3);
        assert!(matches!(ir.ops[0], PcuDispatchOp::GridStrideLoop { .. }));
        assert_eq!(
            description
                .resource(PcuBindingRef::new(0, 1))
                .unwrap()
                .minimum_initial_read_elements,
            0
        );
    });
    let bindings = matrix_bindings::<T>();
    matrix_ir::<T, 2, 3>(&bindings).unwrap().with_ir(|ir| {
        let description = describe_scalar_transport_map::<3>(ir, T::TYPE).unwrap();
        assert_eq!(description.logical_extent, 6);
        assert_eq!(
            description
                .resource(PcuBindingRef::new(0, 2))
                .unwrap()
                .minimum_write_elements,
            6
        );
    });
}

#[test]
fn all_twenty_two_carriers_have_genuine_generic_transport_source() {
    assert_source::<u8>();
    assert_source::<i8>();
    assert_source::<u16>();
    assert_source::<i16>();
    assert_source::<u32>();
    assert_source::<i32>();
    assert_source::<u64>();
    assert_source::<i64>();
    assert_source::<u128>();
    assert_source::<i128>();
    assert_source::<PcuU256>();
    assert_source::<PcuI256>();
    assert_source::<PcuU512>();
    assert_source::<PcuI512>();
    assert_source::<PcuF16Bits>();
    assert_source::<PcuBf16Bits>();
    assert_source::<PcuF8E4M3FnBits>();
    assert_source::<PcuF8E5M2Bits>();
    assert_source::<f32>();
    assert_source::<f64>();
    assert_source::<PcuF128Bits>();
    assert_source::<PcuF256Bits>();
}

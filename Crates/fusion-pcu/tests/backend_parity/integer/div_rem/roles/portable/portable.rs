//! Function-local Portable flags use the same joint fault/publication law as globals.
use super::*;

#[pcu(invocations = N, flag(deterministic))]
fn direct<T: PcuCheckedIntegerDivision, const N: usize>(
    left: &[T; N],
    right: &[T; N],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(left[id], right[id]);
    quotient[id] = q;
    remainder[id] = r;
}

#[pcu(invocations = 3, flag(deterministic))]
fn grid<T: PcuCheckedIntegerDivision, const N: usize>(
    left: &[T; N],
    right: &[T; N],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (q, r) = pcu::checked_div_rem(left[id], right[id]);
        quotient[id] = q;
        remainder[id] = r;
        id += stride;
    }
}

fn format<T: Sample>() {
    global::clear_thread_cache().unwrap();
    super::super::execute::<T>(direct::<T, 7>);
    super::super::execute::<T>(grid::<T, 7>);
}

pub(super) fn verify() {
    format::<u8>();
    format::<i8>();
    format::<u16>();
    format::<i16>();
    format::<u32>();
    format::<i32>();
    format::<u64>();
    format::<i64>();
    format::<u128>();
    format::<i128>();
    format::<PcuU256>();
    format::<PcuI256>();
    format::<PcuU512>();
    format::<PcuI512>();
}

fn assert_local_ir<T: PcuCheckedIntegerDivision>() {
    let direct_bindings = direct_bindings::<T>();
    let grid_bindings = grid_bindings::<T>();
    let direct = direct_ir::<T, 7>(&direct_bindings).unwrap();
    let grid = grid_ir::<T, 7>(&grid_bindings).unwrap();
    for (source, invocations) in [(direct.ir(), 7), (grid.ir(), 3)] {
        assert_eq!(
            source
                .numerical_requirements
                .numerical_options
                .reproducibility,
            PcuReproducibility::PortableV1,
        );
        let description = fusion_pcu::describe_portable_v1_integer_div_rem_map(&source).unwrap();
        assert_eq!(description.scalar, T::TYPE);
        assert_eq!(description.submitted_invocations, invocations);
        assert_eq!(description.logical_extent, 7);
        assert_eq!(description.operands.input_element_counts(7), [7, 7]);
    }
}

#[test]
fn local_portable_division_flags_keep_exact_ir_without_execution() {
    assert_local_ir::<u8>();
    assert_local_ir::<i8>();
    assert_local_ir::<u16>();
    assert_local_ir::<i16>();
    assert_local_ir::<u32>();
    assert_local_ir::<i32>();
    assert_local_ir::<u64>();
    assert_local_ir::<i64>();
    assert_local_ir::<u128>();
    assert_local_ir::<i128>();
    assert_local_ir::<PcuU256>();
    assert_local_ir::<PcuI256>();
    assert_local_ir::<PcuU512>();
    assert_local_ir::<PcuI512>();
}

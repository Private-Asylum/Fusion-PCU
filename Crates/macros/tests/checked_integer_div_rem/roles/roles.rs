//! Actual source uses detached mathematical roles; eligibility is not backend qualification.
use super::*;
#[rustfmt::skip]
use pcu_alias::{
    assess_checked_integer_div_rem_operands,
    PcuCheckedIntegerDivision,
    PcuDispatchIndex as Index,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn repeated<T: PcuCheckedIntegerDivision, const N: usize>(
    q: &mut [T],
    input: &[T; N],
    r: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[id]);
    r[id] = remainder;
    q[id] = quotient;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn unread<T: PcuCheckedIntegerDivision, const N: usize>(
    _unused: &[T],
    r: &mut [T],
    input: &[T; N],
    q: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[0]);
    q[id] = quotient;
    r[id] = remainder;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn reordered<T: PcuCheckedIntegerDivision, const N: usize>(
    right: &[T; N],
    r: &mut [T],
    left: &[T; N],
    q: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(left[id], right[id]);
    r[id] = remainder;
    q[id] = quotient;
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn grid_broadcast<T: PcuCheckedIntegerDivision, const N: usize>(
    r: &mut [T],
    input: &[T; N],
    q: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (quotient, remainder) = pcu::checked_div_rem(input[0], input[id]);
        r[id] = remainder;
        q[id] = quotient;
        id += stride;
    }
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn scalar_divisor<T: PcuCheckedIntegerDivision, const N: usize>(
    q: &mut [T],
    divisor: &T,
    r: &mut [T],
    input: &[T; N],
) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], *divisor);
    q[id] = quotient;
    r[id] = remainder;
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn scalar_grid<T: PcuCheckedIntegerDivision, const N: usize>(input: &T, q: &mut [T], r: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (quotient, remainder) = pcu::checked_div_rem(*input, *input);
        q[id] = quotient;
        r[id] = remainder;
        id += stride;
    }
}

fn verify<T: PcuCheckedIntegerDivision>() {
    let value_type = PcuValueType::Scalar(T::TYPE);
    let caps = PcuValueTypeCaps::for_scalar(T::TYPE);
    macro_rules! check {
        ($function:ident, $bindings:ident, $inputs:expr, $operand:expr, $indices:expr, $counts:expr, $outputs:expr) => {{
            let bindings = $bindings::<T>();
            let builder = $function::<T, 17>(&bindings).unwrap();
            let ir = builder.ir();
            validate_typed_dispatch_value_flow(&ir).unwrap();
            let roles = assess_checked_integer_div_rem_operands(&ir, value_type, caps).unwrap();
            assert_eq!(
                roles.input_bindings(),
                &$inputs.map(|slot| pcu_alias::PcuBindingRef::new(0, slot))
            );
            assert_eq!(roles.operand_inputs(), $operand);
            assert_eq!(roles.operand_indices(), $indices);
            assert_eq!(roles.input_element_counts(17), $counts);
            assert_eq!(
                roles.output_bindings(),
                $outputs.map(|slot| pcu_alias::PcuBindingRef::new(0, slot))
            );
            assert!(validate_integer_checked_div_rem_kernel(&ir, value_type, caps).is_err());
            // The same genuine source has a neutral Portable description. This does not
            // opt a provider into Portable division or weaken its current refusal gate.
            for mode in [
                pcu_alias::PcuNumericalMode::Boundary,
                pcu_alias::PcuNumericalMode::Strict,
            ] {
                for underflow in [
                    pcu_alias::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    pcu_alias::PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    pcu_alias::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    let mut portable = ir;
                    portable.numerical_requirements.numerical_mode = mode;
                    portable.numerical_requirements.float_underflow = underflow;
                    portable
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = pcu_alias::PcuReproducibility::PortableV1;
                    let description =
                        pcu_alias::describe_portable_v1_integer_div_rem_map(&portable).unwrap();
                    assert_eq!(description.scalar, T::TYPE);
                    assert_eq!(description.logical_extent, 17);
                    assert_eq!(description.operands, roles);
                }
            }
        }};
    }
    check!(
        repeated_ir,
        repeated_bindings,
        [1],
        [0, 0],
        [Index::InvocationId; 2],
        [17, 0],
        [0, 2]
    );
    check!(
        unread_ir,
        unread_bindings,
        [2],
        [0, 0],
        [Index::InvocationId, Index::BindingElementZero],
        [17, 0],
        [3, 1]
    );
    check!(
        reordered_ir,
        reordered_bindings,
        [2, 0],
        [0, 1],
        [Index::InvocationId; 2],
        [17, 17],
        [3, 1]
    );
    check!(
        grid_broadcast_ir,
        grid_broadcast_bindings,
        [1],
        [0, 0],
        [Index::BindingElementZero, Index::GridStrideId],
        [17, 0],
        [2, 0]
    );
    check!(
        scalar_divisor_ir,
        scalar_divisor_bindings,
        [3, 1],
        [0, 1],
        [Index::InvocationId, Index::BindingElementZero],
        [17, 1],
        [0, 2]
    );
    check!(
        scalar_grid_ir,
        scalar_grid_bindings,
        [0],
        [0, 0],
        [Index::BindingElementZero; 2],
        [1, 0],
        [1, 2]
    );
}

#[test]
fn six_genuine_role_profiles_keep_exact_ir_for_all_fourteen_integer_types() {
    verify::<u8>();
    verify::<i8>();
    verify::<u16>();
    verify::<i16>();
    verify::<u32>();
    verify::<i32>();
    verify::<u64>();
    verify::<i64>();
    verify::<u128>();
    verify::<i128>();
    verify::<PcuU256>();
    verify::<PcuI256>();
    verify::<PcuU512>();
    verify::<PcuI512>();
}

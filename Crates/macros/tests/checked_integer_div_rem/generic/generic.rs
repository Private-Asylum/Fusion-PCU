//! Stronger sealed bounds describe exact checked division without claiming a device ABI.
use super::*;
#[rustfmt::skip]
use pcu_alias::{
    PcuCheckedIntegerDivision,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn direct<T: PcuCheckedIntegerDivision, const N: usize>(
    lhs: &[T; N],
    rhs: &[T; N],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
    quotient[id] = q;
    remainder[id] = r;
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn grid<T, const N: usize>(lhs: &[T; N], rhs: &[T; N], quotient: &mut [T], remainder: &mut [T])
where
    T: PcuCheckedIntegerDivision,
{
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
        quotient[id] = q;
        remainder[id] = r;
        id += stride;
    }
}

fn verify<T: PcuCheckedIntegerDivision>() {
    let value_type = PcuValueType::Scalar(T::TYPE);
    let caps = PcuValueTypeCaps::for_scalar(T::TYPE);
    let direct_bindings = direct_bindings::<T>();
    let grid_bindings = grid_bindings::<T>();
    let direct = direct_ir::<T, 17>(&direct_bindings).unwrap();
    let grid = grid_ir::<T, 17>(&grid_bindings).unwrap();
    for ir in [direct.ir(), grid.ir()] {
        validate_integer_checked_div_rem_kernel(&ir, value_type, caps).unwrap();
        validate_typed_dispatch_value_flow(&ir).unwrap();
        let body = match ir.ops[0] {
            PcuDispatchOp::GridStrideLoop { extent: 17, body } => body,
            _ => ir.ops,
        };
        assert!(matches!(body[2], PcuDispatchOp::Data(
            PcuDispatchDataOp::CheckedDivRem { value_type: actual, flags, .. }
        ) if actual == value_type && flags == PcuIntegerDivFlags::CHECKED));
    }
}

#[test]
fn all_fourteen_widths_use_typed_joint_checked_results() {
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

#[test]
fn concrete_and_generic_specializations_have_identical_ir() {
    macro_rules! compare {
        ($($module:ident: $ty:ty),+ $(,)?) => {$(
            let concrete_bindings = super::$module::direct_bindings();
            let generic_bindings = direct_bindings::<$ty>();
            let concrete = super::$module::direct_ir::<17>(&concrete_bindings).unwrap();
            let generic = direct_ir::<$ty, 17>(&generic_bindings).unwrap();
            assert_eq!(concrete.ir(), generic.ir());
            let concrete_bindings = super::$module::grid_bindings();
            let generic_bindings = grid_bindings::<$ty>();
            let concrete = super::$module::grid_ir::<17>(&concrete_bindings).unwrap();
            let generic = grid_ir::<$ty, 17>(&generic_bindings).unwrap();
            assert_eq!(concrete.ir(), generic.ir());
        )+};
    }
    compare!(unsigned8: u8, signed8: i8, unsigned16: u16, signed16: i16,
        unsigned32: u32, signed32: i32, unsigned64: u64, signed64: i64,
        unsigned128: u128, signed128: i128);
}

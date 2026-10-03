//! Source vocabulary is independent of a backend's admitted integer division widths.
#[path = "checked_integer_div_rem/generic/generic.rs"]
mod generic;
#[path = "checked_integer_div_rem/roles/roles.rs"]
mod roles;
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    PcuDispatchDataOp,
    PcuDispatchOp,
    PcuScalar,
    PcuValueType,
    PcuValueTypeCaps,
    model::PcuIntegerDivFlags,
    validate_integer_checked_div_rem_kernel,
    validate_typed_dispatch_value_flow,
};

macro_rules! profiles {
    ($($module:ident: $ty:ty),* $(,)?) => {$(
        mod $module {
            use super::*;
            #[pcu(invocations = N, crate_path = ::pcu_alias)]
            pub fn direct<const N: usize>(lhs: &[$ty; N], rhs: &[$ty; N], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
                quotient[id] = q;
                remainder[id] = r;
            }
            #[pcu(invocations = 3, crate_path = ::pcu_alias)]
            pub fn grid<const N: usize>(lhs: &[$ty; N], rhs: &[$ty; N], quotient: &mut [$ty], remainder: &mut [$ty]) {
                let mut id = pcu::context::global_invocation_id();
                let stride = pcu::context::invocation_count();
                while id < N {
                    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
                    quotient[id] = q;
                    remainder[id] = r;
                    id += stride;
                }
            }
            #[test]
            fn direct_and_grid_keep_exact_types_and_checked_fault_contract() {
                let value_type = PcuValueType::Scalar(<$ty as PcuScalar>::TYPE);
                let caps = PcuValueTypeCaps::for_scalar(<$ty as PcuScalar>::TYPE);
                let bindings = direct_bindings();
                let builder = direct_ir::<17>(&bindings).unwrap();
                let ir = builder.ir();
                validate_integer_checked_div_rem_kernel(&ir, value_type, caps).unwrap();
                validate_typed_dispatch_value_flow(&ir).unwrap();
                assert!(ir.ops.iter().any(|op| matches!(op,
                    PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { value_type: actual, flags: PcuIntegerDivFlags::CHECKED, .. }) if *actual == value_type
                )));
                let bindings = grid_bindings();
                let builder = grid_ir::<17>(&bindings).unwrap();
                let ir = builder.ir();
                validate_integer_checked_div_rem_kernel(&ir, value_type, caps).unwrap();
                validate_typed_dispatch_value_flow(&ir).unwrap();
                assert!(matches!(ir.ops[0], PcuDispatchOp::GridStrideLoop { extent: 17, .. }));
            }
        }
    )*};
}
profiles!(unsigned8: u8, signed8: i8, unsigned16: u16, signed16: i16, unsigned32: u32, signed32: i32, unsigned64: u64, signed64: i64, unsigned128: u128, signed128: i128);

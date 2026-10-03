//! Compound local assignment is the same ordered, checked SSA as explicit reassignment.
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    PcuCheckedFloat,
    PcuCheckedInteger,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};

macro_rules! sources {
    ($bound:ident, $direct:ident, $flat:ident, $grid:ident, $flat_grid:ident) => {
        #[pcu(invocations = N, crate_path = ::pcu_alias)]
        fn $direct<T: $bound, const N: usize>(input: &[T], output: &mut [T]) {
            let id = pcu::context::global_invocation_id();
            let mut value: T = input[id];
            let saved = value;
            value += saved;
            value -= saved;
            value *= saved;
            output[id] = value;
        }
        #[pcu(invocations = N, crate_path = ::pcu_alias)]
        fn $flat<T: $bound, const N: usize>(input: &[T], output: &mut [T]) {
            let id = pcu::context::global_invocation_id();
            let mut value: T = input[id];
            let saved = value;
            value = value + saved;
            value = value - saved;
            value = value * saved;
            output[id] = value;
        }
        #[pcu(invocations = 3, crate_path = ::pcu_alias)]
        fn $grid<T: $bound, const N: usize>(input: &[T], output: &mut [T]) {
            let mut id = pcu::context::global_invocation_id();
            let stride = pcu::context::invocation_count();
            while id < N {
                let mut value: T = input[id];
                let saved = value;
                value += saved;
                value -= saved;
                value *= saved;
                output[id] = value;
                id += stride;
            }
        }
        #[pcu(invocations = 3, crate_path = ::pcu_alias)]
        fn $flat_grid<T: $bound, const N: usize>(input: &[T], output: &mut [T]) {
            let mut id = pcu::context::global_invocation_id();
            let stride = pcu::context::invocation_count();
            while id < N {
                let mut value: T = input[id];
                let saved = value;
                value = value + saved;
                value = value - saved;
                value = value * saved;
                output[id] = value;
                id += stride;
            }
        }
    };
}
sources!(
    PcuCheckedFloat,
    floats,
    flat_floats,
    grid_floats,
    flat_grid_floats
);
sources!(
    PcuCheckedInteger,
    integers,
    flat_integers,
    grid_integers,
    flat_grid_integers
);

macro_rules! compare {
    ($bindings:ident, $ir:ident, $flat_bindings:ident, $flat_ir:ident, $ty:ty) => {
        for request in super::typed::requests() {
            let bindings = $bindings::<$ty>();
            let flat_bindings = $flat_bindings::<$ty>();
            $ir::<$ty, 47>(
                &bindings,
                request.float_underflow,
                request.range_policy,
                request,
            )
            .unwrap()
            .with_ir(|source| {
                $flat_ir::<$ty, 47>(
                    &flat_bindings,
                    request.float_underflow,
                    request.range_policy,
                    request,
                )
                .unwrap()
                .with_ir(|flat| {
                    assert_eq!(source.ops, flat.ops);
                    assert_eq!(source.bindings, flat.bindings);
                    assert_eq!(source.numerical_requirements, flat.numerical_requirements);
                    pcu_alias::validate_typed_dispatch_value_flow(source).unwrap();
                });
            });
        }
    };
}

#[test]
fn six_floats_and_fourteen_integers_keep_checked_compound_assignment_ir() {
    macro_rules! floats { ($($ty:ty),+) => { $(
        compare!(floats_bindings, __floats_ir_with_float_underflow_policy, flat_floats_bindings, __flat_floats_ir_with_float_underflow_policy, $ty);
        compare!(grid_floats_bindings, __grid_floats_ir_with_float_underflow_policy, flat_grid_floats_bindings, __flat_grid_floats_ir_with_float_underflow_policy, $ty);
    )+ }; }
    macro_rules! integers { ($($ty:ty),+) => { $(
        compare!(integers_bindings, __integers_ir_with_float_underflow_policy, flat_integers_bindings, __flat_integers_ir_with_float_underflow_policy, $ty);
        compare!(grid_integers_bindings, __grid_integers_ir_with_float_underflow_policy, flat_grid_integers_bindings, __flat_grid_integers_ir_with_float_underflow_policy, $ty);
    )+ }; }
    floats!(
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64
    );
    integers!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

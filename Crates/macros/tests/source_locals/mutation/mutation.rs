//! Actual generic source keeps old aliases while reassignment produces fresh SSA.
use super::*;

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn float_mutation<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let mut value = input[id];
    let original = value;
    value = value + value;
    value = value * original;
    output[id] = value;
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn grid_mutation<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let mut value = input[id];
        let original = value;
        value = value + value;
        value = value * original;
        output[id] = value;
        id += stride;
    }
}

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn matrix_mutation<T: PcuCheckedFloat, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    let row = id / C;
    let col = id % C;
    let mut value = input[row][col];
    let original = value;
    value = value + value;
    value = value * original;
    output[row][col] = value;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn integer_mutation<T: PcuCheckedInteger, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let mut value = input[id];
    let original = value;
    value = value + value;
    value = value * original;
    output[id] = value;
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn discarded_mutation<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    denominator: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let mut value = input[id];
    value = value / denominator[id];
    value = input[id];
    output[id] = value;
}

fn check_float(ops: &[Op<'_>]) {
    assert!(matches!(
        ops[0],
        Op::Data(Data::BindingLoad { result: Id(1), .. })
    ));
    assert!(matches!(
        ops[1],
        Op::Data(Data::CheckedFloatBinary {
            op: FloatOp::Add,
            lhs: Id(1),
            rhs: Id(1),
            result: Id(2),
            ..
        })
    ));
    assert!(matches!(
        ops[2],
        Op::Data(Data::CheckedFloatBinary {
            op: FloatOp::Mul,
            lhs: Id(2),
            rhs: Id(1),
            result: Id(3),
            ..
        })
    ));
    assert!(matches!(
        ops[3],
        Op::Data(Data::BindingStore { value: Id(3), .. })
    ));
    assert_eq!(ops.len(), 4);
}

fn floating<T: PcuCheckedFloat>() {
    let bindings = float_mutation_bindings::<T>();
    float_mutation_ir::<T, 7>(&bindings).unwrap().with_ir(|ir| {
        validate_typed_dispatch_value_flow(ir).unwrap();
        assert_eq!(ir.ops.len(), 5);
        check_float(&ir.ops[..4]);
    });
    let bindings = grid_mutation_bindings::<T>();
    grid_mutation_ir::<T, 19>(&bindings).unwrap().with_ir(|ir| {
        validate_typed_dispatch_value_flow(ir).unwrap();
        let Op::GridStrideLoop { body, .. } = ir.ops[0] else {
            panic!("canonical grid body")
        };
        check_float(body);
    });
    let bindings = matrix_mutation_bindings::<T>();
    matrix_mutation_ir::<T, 2, 3>(&bindings)
        .unwrap()
        .with_ir(|ir| {
            validate_typed_dispatch_value_flow(ir).unwrap();
            assert_eq!(ir.ops.len(), 5);
            check_float(&ir.ops[..4]);
        });
}

#[test]
fn all_six_floats_keep_original_aliases_in_direct_grid_and_matrix_source() {
    floating::<PcuF16Bits>();
    floating::<PcuBf16Bits>();
    floating::<PcuF8E4M3FnBits>();
    floating::<PcuF8E5M2Bits>();
    floating::<f32>();
    floating::<f64>();
}

fn integer<T: PcuCheckedInteger>() {
    let bindings = integer_mutation_bindings::<T>();
    integer_mutation_ir::<T, 7>(&bindings)
        .unwrap()
        .with_ir(|ir| {
            validate_typed_dispatch_value_flow(ir).unwrap();
            assert_eq!(ir.ops.len(), 5);
            assert!(matches!(
                ir.ops[1],
                Op::Data(Data::CheckedIntegerBinary {
                    op: IntegerOp::Add,
                    lhs: Id(1),
                    rhs: Id(1),
                    result: Id(2),
                    ..
                })
            ));
            assert!(matches!(
                ir.ops[2],
                Op::Data(Data::CheckedIntegerBinary {
                    op: IntegerOp::Mul,
                    lhs: Id(2),
                    rhs: Id(1),
                    result: Id(3),
                    ..
                })
            ));
        });
}

#[test]
fn all_fourteen_integer_widths_keep_original_aliases() {
    integer::<i8>();
    integer::<u8>();
    integer::<i16>();
    integer::<u16>();
    integer::<i32>();
    integer::<u32>();
    integer::<i64>();
    integer::<u64>();
    integer::<i128>();
    integer::<u128>();
    integer::<PcuI256>();
    integer::<PcuU256>();
    integer::<PcuI512>();
    integer::<PcuU512>();
}

#[test]
fn overwritten_checked_value_keeps_its_observable_fault_operation() {
    let bindings = discarded_mutation_bindings::<f32>();
    discarded_mutation_ir::<f32, 7>(&bindings)
        .unwrap()
        .with_ir(|ir| {
            validate_typed_dispatch_value_flow(ir).unwrap();
            assert_eq!(ir.ops.len(), 6);
            assert!(matches!(
                ir.ops[2],
                Op::Data(Data::CheckedFloatBinary {
                    op: FloatOp::Div,
                    result: Id(3),
                    ..
                })
            ));
            assert!(matches!(
                ir.ops[3],
                Op::Data(Data::BindingLoad { result: Id(4), .. })
            ));
            assert!(matches!(
                ir.ops[4],
                Op::Data(Data::BindingStore { value: Id(4), .. })
            ));
        });
}

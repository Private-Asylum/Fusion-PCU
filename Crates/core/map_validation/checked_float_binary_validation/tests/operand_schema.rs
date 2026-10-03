//! Actual SSA roles must survive declaration reordering, repeated loads and broadcast.

#[rustfmt::skip]
use super::{
    body,
    fixture,
    make_bindings,
    validate,
    BinaryOp,
    Data,
    Error,
    Id,
    Op,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchIndex,
    PcuFloatUnderflowPolicy,
    PcuValueType,
    PcuValueTypeCaps,
};
use crate::assess_checked_float_binary_operands;

#[test]
fn declaration_order_and_mathematical_order_are_independent() {
    let original = make_bindings(PcuValueType::f32());
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let bindings = order.map(|slot| original[slot]);
        for (lhs, rhs, expected) in [
            (Id(1), Id(2), [0, 1]),
            (Id(2), Id(1), [1, 0]),
            (Id(1), Id(1), [0, 0]),
        ] {
            let mut instructions = body(
                PcuDispatchIndex::InvocationId,
                Id(3),
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
            );
            let Op::Data(Data::CheckedFloatBinary {
                lhs: left,
                rhs: right,
                ..
            }) = &mut instructions[2]
            else {
                unreachable!()
            };
            *left = lhs;
            *right = rhs;
            let ops = [
                instructions[0],
                instructions[1],
                instructions[2],
                instructions[3],
                Op::Control(PcuDispatchControlOp::Return),
            ];
            let schema = assess_checked_float_binary_operands(
                &fixture(&bindings, &ops, &[]),
                PcuValueType::f32(),
                BinaryOp::Add,
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuValueTypeCaps::FLOAT32,
            )
            .unwrap();
            assert_eq!(
                schema.input_bindings(),
                &[original[0].reference(), original[1].reference()]
            );
            assert_eq!(schema.operand_inputs(), expected);
            assert_eq!(schema.input_element_counts(19), [19, 19]);
            assert_eq!(schema.output_binding(), original[2].reference());
        }
    }
}

#[test]
fn repeated_binding_retains_independent_indices_and_one_input_extent() {
    let declared = make_bindings(PcuValueType::f32());
    for grid in [false, true] {
        let index = if grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        for scalar_first in [false, true] {
            let mut instructions = body(index, Id(3), PcuFloatUnderflowPolicy::IeeeAfterRounding);
            let Op::Data(Data::BindingLoad {
                binding,
                index: second,
                ..
            }) = &mut instructions[1]
            else {
                unreachable!()
            };
            *binding = declared[0].reference();
            *second = if scalar_first {
                index
            } else {
                PcuDispatchIndex::BindingElementZero
            };
            if scalar_first {
                let Op::Data(Data::BindingLoad { index: first, .. }) = &mut instructions[0] else {
                    unreachable!()
                };
                *first = PcuDispatchIndex::BindingElementZero;
            }
            let direct = [
                instructions[0],
                instructions[1],
                instructions[2],
                instructions[3],
                Op::Control(PcuDispatchControlOp::Return),
            ];
            let grid_ops = [
                Op::GridStrideLoop {
                    extent: 19,
                    body: &instructions,
                },
                Op::Control(PcuDispatchControlOp::Return),
            ];
            let ops = if grid { &grid_ops[..] } else { &direct[..] };
            // The same actual reads are valid with or without an unused Rust parameter.
            let single = [declared[2], declared[0]];
            for bindings in [&declared[..], &single[..]] {
                let schema = assess_checked_float_binary_operands(
                    &fixture(bindings, ops, &[]),
                    PcuValueType::f32(),
                    BinaryOp::Add,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuValueTypeCaps::FLOAT32,
                )
                .unwrap();
                assert_eq!(schema.input_bindings(), &[declared[0].reference()]);
                assert_eq!(schema.operand_inputs(), [0, 0]);
                assert_eq!(schema.input_element_counts(19), [19, 0]);
                assert_eq!(
                    schema.operand_indices(),
                    if scalar_first {
                        [PcuDispatchIndex::BindingElementZero, index]
                    } else {
                        [index, PcuDispatchIndex::BindingElementZero]
                    }
                );
            }
            assert!(
                validate(
                    &fixture(&declared, ops, &[]),
                    BinaryOp::Add,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuValueTypeCaps::FLOAT32
                )
                .is_err()
            );
        }
    }
}

#[test]
fn repeated_scalar_reads_need_only_one_element_and_bad_ssa_still_fails() {
    let bindings = make_bindings(PcuValueType::f32());
    let mut instructions = body(
        PcuDispatchIndex::InvocationId,
        Id(3),
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    );
    for instruction in &mut instructions[..2] {
        let Op::Data(Data::BindingLoad { binding, index, .. }) = instruction else {
            unreachable!()
        };
        *binding = PcuBindingRef::new(0, 0);
        *index = PcuDispatchIndex::BindingElementZero;
    }
    let mut ops = [
        instructions[0],
        instructions[1],
        instructions[2],
        instructions[3],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let assess = |ops: &[Op<'_>]| {
        assess_checked_float_binary_operands(
            &fixture(&bindings, ops, &[]),
            PcuValueType::f32(),
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuValueTypeCaps::FLOAT32,
        )
    };
    assert_eq!(assess(&ops).unwrap().input_element_counts(19), [1, 0]);
    let Op::Data(Data::BindingLoad { result, .. }) = &mut ops[1] else {
        unreachable!()
    };
    *result = Id(1);
    assert_eq!(assess(&ops), Err(Error::InvalidSsa));
}

#[test]
fn writable_loads_duplicate_bindings_and_multiple_outputs_remain_invalid() {
    let original = make_bindings(PcuValueType::f32());
    let instructions = body(
        PcuDispatchIndex::InvocationId,
        Id(3),
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    );
    let ops = [
        instructions[0],
        instructions[1],
        instructions[2],
        instructions[3],
        Op::Control(PcuDispatchControlOp::Return),
    ];
    let assess = |bindings: &[crate::PcuBinding<'_>]| {
        assess_checked_float_binary_operands(
            &fixture(bindings, &ops, &[]),
            PcuValueType::f32(),
            BinaryOp::Add,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuValueTypeCaps::FLOAT32,
        )
    };
    let mut bindings = original;
    bindings[1] = original[0];
    assert_eq!(
        assess(&bindings),
        Err(Error::DuplicateBinding(original[0].reference()))
    );
    bindings = original;
    bindings[1].access = PcuBindingAccess::ReadWrite;
    assert!(assess(&bindings).is_err());
    assert!(assess(&original[1..]).is_err());
}

//! Cold unique resource, declaration and exact original request proof across fourteen widths.
#[rustfmt::skip]
use super::{
    tests::fixture_typed,
    Profile,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuRangePolicy,
    PcuScalar,
    PcuImplementationRequirements,
};
fn complete<T: PcuScalar>() {
    for grid in [false, true] {
        for op in [
            PcuDispatchIntegerBinaryOp::Add,
            PcuDispatchIntegerBinaryOp::Sub,
            PcuDispatchIntegerBinaryOp::Mul,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                    for compound in [
                        PcuCompoundArithmeticPolicy::Checked,
                        PcuCompoundArithmeticPolicy::BackendDefined,
                    ] {
                        for precision in [
                            PcuPrecisionPolicy::Preserve,
                            PcuPrecisionPolicy::BackendOptimized,
                        ] {
                            for underflow in [
                                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                            ] {
                                for (single, zero_first, zero_second, repeat_math) in [
                                    (false, false, false, false),
                                    (true, false, false, false),
                                    (false, true, false, false),
                                    (true, true, true, false),
                                    (false, false, false, true),
                                ] {
                                    fixture_typed::<T>(grid, false, |original| {
                                        let mut requirements = original.numerical_requirements;
                                        requirements.range_policy = range;
                                        requirements.numerical_mode = mode;
                                        requirements.numerical_options.compound_arithmetic =
                                            compound;
                                        requirements.numerical_options.precision = precision;
                                        requirements.float_underflow = underflow;
                                        verify::<T>(
                                            original,
                                            grid,
                                            op,
                                            requirements,
                                            (single, zero_first, zero_second, repeat_math),
                                        );
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
fn verify<T: PcuScalar>(
    original: &PcuDispatchKernelIr<'_>,
    grid: bool,
    operation: PcuDispatchIntegerBinaryOp,
    requirements: PcuImplementationRequirements,
    shape: (bool, bool, bool, bool),
) {
    let (single, zero_first, zero_second, repeat_math) = shape;
    let first = original.bindings[0];
    let unused = original.bindings[1];
    let output = original.bindings[2];
    let declarations = if single {
        vec![output, first]
    } else {
        vec![output, unused, first]
    };
    let mut body = match original.ops[0] {
        PcuDispatchOp::GridStrideLoop { body, .. } => body.to_vec(),
        _ => original.ops[..4].to_vec(),
    };
    for (slot, zero) in [zero_first, zero_second].into_iter().enumerate() {
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. }) =
            &mut body[slot]
        else {
            panic!("load");
        };
        *binding = first.reference();
        if zero {
            *index = PcuDispatchIndex::BindingElementZero;
        }
    }
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: first_value,
        ..
    }) = body[0]
    else {
        panic!("load value");
    };
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
        op,
        range_policy,
        lhs,
        rhs,
        ..
    }) = &mut body[2]
    else {
        panic!("binary");
    };
    *op = operation;
    *range_policy = requirements.range_policy;
    if repeat_math {
        *lhs = first_value;
        *rhs = first_value;
    }
    let mut direct = body.clone();
    direct.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
    let grid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 3,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut kernel = *original;
    kernel.bindings = &declarations;
    kernel.ops = if grid { &grid_ops } else { &direct };
    kernel.numerical_requirements = requirements;
    let profile = Profile::admit(&kernel).unwrap();
    assert_eq!(profile.read_count, 1);
    assert_eq!(profile.loads[0], first.reference());
    assert_eq!(profile.inputs, [first.reference(); 2]);
    assert_eq!(profile.operand_slots, [0, 0]);
    assert_eq!(profile.output, output.reference());
    assert_eq!(
        profile.broadcast,
        if repeat_math {
            [zero_first; 2]
        } else {
            [zero_second, zero_first]
        }
    );
    let width = usize::from(T::TYPE.bit_width()) / 8;
    assert_eq!(
        profile.read_bytes,
        [
            if zero_first && zero_second {
                width
            } else {
                width * 3
            },
            0
        ]
    );
    let mut denied = declarations.clone();
    denied[if single { 1 } else { 2 }].access = PcuBindingAccess::ReadWrite;
    kernel.bindings = &denied;
    assert!(Profile::admit(&kernel).is_err());
    kernel.bindings = &declarations;
    kernel
        .numerical_requirements
        .numerical_options
        .reproducibility = fusion_pcu::PcuReproducibility::PortableV1;
    assert!(Profile::admit(&kernel).is_err());
}
#[test]
fn fourteen_widths_repeated_reordered_single_declaration_and_independent_index_roles() {
    complete::<i8>();
    complete::<u8>();
    complete::<i16>();
    complete::<u16>();
    complete::<i32>();
    complete::<u32>();
    complete::<i64>();
    complete::<u64>();
    complete::<i128>();
    complete::<u128>();
    complete::<fusion_pcu::PcuI256>();
    complete::<fusion_pcu::PcuU256>();
    complete::<fusion_pcu::PcuI512>();
    complete::<fusion_pcu::PcuU512>();
}

//! Original declarations, actual read spans and operand slots are independent cold facts.
#[rustfmt::skip]
use super::{
    fixture,
    Op,
    Policy,
    Profile,
    Variant,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuNumericalMode,
    PcuDispatchKernelIr,
    PcuRangePolicy,
    PcuScalar,
};

fn complete<T: PcuScalar>() {
    for grid in [false, true] {
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
            for policy in [
                Policy::IeeeAfterRounding,
                Policy::AllowGradualUnderflow,
                Policy::RejectSubnormalResult,
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
                                for (one_declaration, zero_first, zero_second) in [
                                    (false, false, false),
                                    (true, false, false),
                                    (false, true, false),
                                    (true, true, true),
                                ] {
                                    fixture::<T>(grid, op, policy, Variant::Plain, |original| {
                                        verify::<T>(
                                            original,
                                            grid,
                                            range,
                                            mode,
                                            compound,
                                            precision,
                                            (one_declaration, zero_first, zero_second),
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
    range: PcuRangePolicy,
    mode: PcuNumericalMode,
    compound: PcuCompoundArithmeticPolicy,
    precision: PcuPrecisionPolicy,
    shape: (bool, bool, bool),
) {
    let (one_declaration, zero_first, zero_second) = shape;
    let first = original.bindings[0];
    let unused = original.bindings[1];
    let output = original.bindings[2];
    let declarations = if one_declaration {
        vec![output, first]
    } else {
        vec![output, unused, first]
    };
    let original_body = match original.ops[0] {
        PcuDispatchOp::GridStrideLoop { body, .. } => body,
        _ => &original.ops[..4],
    };
    let mut body = original_body.to_vec();
    for (slot, zero) in [zero_first, zero_second].into_iter().enumerate() {
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. }) =
            &mut body[slot]
        else {
            panic!("load fixture");
        };
        *binding = first.reference();
        if zero {
            *index = PcuDispatchIndex::BindingElementZero;
        }
    }
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { range_policy, .. }) =
        &mut body[2]
    else {
        panic!("binary fixture");
    };
    *range_policy = range;
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
    kernel.numerical_requirements.range_policy = range;
    kernel.numerical_requirements.numerical_mode = mode;
    kernel
        .numerical_requirements
        .numerical_options
        .compound_arithmetic = compound;
    kernel.numerical_requirements.numerical_options.precision = precision;
    let profile = Profile::admit(&kernel).unwrap();
    assert_eq!(profile.requirements, kernel.numerical_requirements);
    assert_eq!(profile.read_count, 1);
    assert_eq!(profile.loads[0], first.reference());
    assert_eq!(profile.inputs, [first.reference(); 2]);
    assert_eq!(profile.operand_slots, [0, 0]);
    assert_eq!(profile.output, output.reference());
    assert_eq!(profile.broadcast, [zero_first, zero_second]);
    let width = usize::from(T::TYPE.bit_width()) / 8;
    assert_eq!(
        profile.input_bytes,
        [
            if zero_first && zero_second {
                width
            } else {
                width * 3
            },
            0
        ]
    );
    if !one_declaration {
        let mut invalid_declarations = declarations.clone();
        invalid_declarations[1].access = PcuBindingAccess::ReadWrite;
        kernel.bindings = &invalid_declarations;
        assert!(Profile::admit(&kernel).is_err());
    }
}

#[test]
fn six_formats_repeated_reordered_single_declaration_and_independent_index_roles() {
    complete::<f32>();
    complete::<f64>();
    complete::<fusion_pcu::PcuF16Bits>();
    complete::<fusion_pcu::PcuBf16Bits>();
    complete::<fusion_pcu::PcuF8E4M3FnBits>();
    complete::<fusion_pcu::PcuF8E5M2Bits>();
}

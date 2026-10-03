//! Detached normal header and actual unary resource-role qualification.
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalCheckedUnaryPlan,
    MetalPortableUnaryPlan,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchFloatUnaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy as Range,
    PcuReproducibility,
    PcuScalar,
    PcuBf16Bits,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
use super::portable_unary::graph;
fn roles<T: PcuScalar>() {
    for op in [Op::Neg, Op::Relu] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::AllowGradualUnderflow,
            Policy::RejectSubnormalResult,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        graph::fixture_profile::<T, _>(
                            5,
                            op,
                            policy,
                            range,
                            grid,
                            broadcast,
                            |ir| {
                                let unused = PcuBinding::scalar::<T>(
                                    Some("unread"),
                                    7,
                                    2,
                                    PcuBindingStorageClass::Storage,
                                    PcuBindingAccess::ReadOnly,
                                );
                                let declarations = [ir.bindings[1], unused, ir.bindings[0]];
                                let mut request = *ir;
                                request.bindings = &declarations;
                                for flags in 0..8 {
                                    request.numerical_requirements.numerical_mode =
                                        if flags & 1 == 0 {
                                            PcuNumericalMode::Boundary
                                        } else {
                                            PcuNumericalMode::Strict
                                        };
                                    request
                                        .numerical_requirements
                                        .numerical_options
                                        .compound_arithmetic = if flags & 2 == 0 {
                                        PcuCompoundArithmeticPolicy::Checked
                                    } else {
                                        PcuCompoundArithmeticPolicy::BackendDefined
                                    };
                                    request.numerical_requirements.numerical_options.precision =
                                        if flags & 4 == 0 {
                                            PcuPrecisionPolicy::Preserve
                                        } else {
                                            PcuPrecisionPolicy::BackendOptimized
                                        };
                                    assert_eq!(
                                        request
                                            .numerical_requirements
                                            .numerical_options
                                            .reproducibility,
                                        PcuReproducibility::Unspecified
                                    );
                                    let plan = MetalCheckedUnaryPlan::assess(&request).unwrap();
                                    assert_eq!(
                                        plan.description().input_binding,
                                        ir.bindings[0].reference()
                                    );
                                    assert_eq!(
                                        plan.description().output_binding,
                                        ir.bindings[1].reference()
                                    );
                                    assert_eq!(plan.requirements(), request.numerical_requirements);
                                    assert!(MetalPortableUnaryPlan::assess(&request).is_err());
                                }
                            },
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn six_format_normal_unary_actual_roles_and_original_headers_are_detached() {
    roles::<PcuF16Bits>();
    roles::<PcuBf16Bits>();
    roles::<PcuF8E4M3FnBits>();
    roles::<PcuF8E5M2Bits>();
    roles::<f32>();
    roles::<f64>();
}

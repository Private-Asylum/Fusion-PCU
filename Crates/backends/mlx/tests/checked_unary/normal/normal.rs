//! Original normal headers, actual unary roles and unread declarations without dummy resources.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedUnaryPlan,
    MlxRuntime,
};
#[rustfmt::skip]
use pcu_facade::{
    describe_portable_v1_unary_map,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchFloatUnaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuRangePolicy as Range,
    PcuReproducibility,
    PcuBf16Bits,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[rustfmt::skip]
use super::{
    graph,
    prefix::Sample,
    portable::native_case,
};
fn roles<T: Sample>(native: bool) {
    let session = native.then(|| MlxRuntime::load_default().unwrap().open_gpu(0).unwrap());
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
                                    assert!(describe_portable_v1_unary_map(&request).is_err());
                                    let plan = MlxCheckedUnaryPlan::assess(&request).unwrap();
                                    assert_eq!(plan.input_bindings(), &[PcuBindingRef::new(2, 3)]);
                                    assert_eq!(plan.requirements(), request.numerical_requirements);
                                    if let Some(session) = &session {
                                        let mut prepared =
                                            session.prepare_unary_host_kernel(&request).unwrap();
                                        assert_eq!(
                                            prepared.requirements(),
                                            request.numerical_requirements
                                        );
                                        assert_eq!(
                                            prepared.unused_bindings(),
                                            &[unused.reference()]
                                        );
                                        native_case::<T>(
                                            &mut prepared,
                                            op,
                                            policy,
                                            range,
                                            broadcast,
                                        );
                                    }
                                }
                            },
                        );
                    }
                }
            }
        }
    }
}
macro_rules! six {
    ($native:expr) => {
        roles::<PcuF16Bits>($native);
        roles::<PcuBf16Bits>($native);
        roles::<PcuF8E4M3FnBits>($native);
        roles::<PcuF8E5M2Bits>($native);
        roles::<f32>($native);
        roles::<f64>($native);
    };
}
#[test]
fn six_format_normal_unary_actual_roles_and_original_headers_are_detached() {
    six!(false);
}
#[test]
#[ignore = "Requires actual MLX original normal unary headers, reordered and unread declarations, exact faults and terminal host publication."]
fn six_format_normal_unary_roles_original_headers_and_publication() {
    six!(true);
}

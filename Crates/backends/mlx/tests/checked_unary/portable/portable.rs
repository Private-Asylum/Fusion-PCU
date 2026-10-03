//! Requested Portable headers retain actual roles and all numerical axes.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedUnaryPlan,
    MlxRuntime,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCompoundArithmeticPolicy,
    PcuDispatchFloatUnaryOp as Op,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy as Policy,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuPreparedHostKernel,
    PcuRangePolicy as Range,
    PcuReproducibility,
};
#[rustfmt::skip]
use super::{
    graph,
    prefix::{
        compare,
        reference,
        Sample,
    },
};
#[path = "encodings/encodings.rs"]
mod encodings;

fn schema<T: Sample>(native: bool) {
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
                                request
                                    .numerical_requirements
                                    .numerical_options
                                    .reproducibility = PcuReproducibility::PortableV1;
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
                                    native_case::<T>(&mut prepared, op, policy, range, broadcast);
                                }
                            },
                        );
                    }
                }
            }
        }
    }
}
fn tuples<T: Sample>() {
    graph::fixture_profile::<T, _>(
        5,
        Op::Neg,
        Policy::IeeeAfterRounding,
        Range::Reject,
        false,
        false,
        |ir| {
            for bits in 0..8 {
                let mut request = *ir;
                request.numerical_requirements.numerical_mode = if bits & 1 == 0 {
                    PcuNumericalMode::Boundary
                } else {
                    PcuNumericalMode::Strict
                };
                request
                    .numerical_requirements
                    .numerical_options
                    .compound_arithmetic = if bits & 2 == 0 {
                    PcuCompoundArithmeticPolicy::Checked
                } else {
                    PcuCompoundArithmeticPolicy::BackendDefined
                };
                request.numerical_requirements.numerical_options.precision = if bits & 4 == 0 {
                    PcuPrecisionPolicy::Preserve
                } else {
                    PcuPrecisionPolicy::BackendOptimized
                };
                request
                    .numerical_requirements
                    .numerical_options
                    .reproducibility = PcuReproducibility::PortableV1;
                assert_eq!(
                    MlxCheckedUnaryPlan::assess(&request)
                        .unwrap()
                        .requirements(),
                    request.numerical_requirements
                );
                request.numerical_requirements.float_underflow = Policy::RejectSubnormalResult;
                assert!(MlxCheckedUnaryPlan::assess(&request).is_err());
            }
        },
    );
}
macro_rules! six {
    ($function:ident, $($argument:expr),*) => {
        $function::<PcuF16Bits>($($argument),*);
        $function::<PcuBf16Bits>($($argument),*);
        $function::<PcuF8E4M3FnBits>($($argument),*);
        $function::<PcuF8E5M2Bits>($($argument),*);
        $function::<f32>($($argument),*);
        $function::<f64>($($argument),*);
    };
}
#[test]
fn six_format_portable_unary_detached_roles_and_original_tuple() {
    six!(schema, false);
    six!(tuples,);
}
#[test]
#[ignore = "Requires actual MLX Portable headers, reordered and unused declarations, exact status and host publication."]
fn six_format_portable_unary_reordered_unused_host_publication() {
    six!(schema, true);
}

pub fn native_case<T: Sample>(
    prepared: &mut fusion_pcu_mlx::MlxPreparedHostKernel,
    op: Op,
    policy: Policy,
    range: Range,
    broadcast: bool,
) {
    assert_eq!(prepared.output_binding(), PcuBindingRef::new(4, 1));
    let input = [
        T::raw(1),
        T::raw(T::SIGN),
        T::raw(T::MAX),
        T::raw(T::SIGN | 1),
        T::raw(0),
    ];
    let sentinel = T::raw(17);
    let mut output = [sentinel; 8];
    let expected = reference(&input, op, policy, range, broadcast);
    let actual = prepared.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(7, 2), &[] as &[T]),
        PcuHostArgument::read_write(prepared.output_binding(), &mut output),
        PcuHostArgument::read(prepared.input_binding(), &input),
    ]);
    if let Some(fault) = expected.1 {
        assert!(
            matches!(actual, Err(PcuHostDispatchError::Backend(fusion_pcu_mlx::MlxError::Arithmetic(actual))) if actual==fault)
        );
    } else {
        actual.unwrap();
    }
    if expected.1.is_none_or(|fault| fault.recovered) {
        compare(&output[..5], &expected.0);
    } else {
        compare(&output[..5], &[sentinel; 5]);
    }
    compare(&output[5..], &[sentinel; 3]);
    let bad = prepared.call(&mut [PcuHostArgument::read_write(
        PcuBindingRef::new(7, 2),
        &mut output,
    )]);
    assert!(
        matches!(bad, Err(PcuHostDispatchError::AccessMismatch(binding)) if binding==PcuBindingRef::new(7, 2))
    );
    assert!(!prepared.last_call_may_have_written());
}

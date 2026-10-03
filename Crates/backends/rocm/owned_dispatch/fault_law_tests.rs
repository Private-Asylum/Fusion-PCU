//! Packed scalar status admission is distinct from private compound event ordinals.
#[path = "../benches/joint_div_rem_operands/source/source.rs"]
#[allow(dead_code)] // Genuine source is retained solely to derive independent cold IR fixtures.
pub mod source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuDispatchCheckedFloatConversion,
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIntegerBinaryOp,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuRangePolicy,
    PcuScalarType,
};
fn valid(tag: u64, recovered: bool, law: PcuCheckedScalarFaultLaw) -> bool {
    super::decode_fault_word_under_law(
        (2 << 3) | tag | if recovered { 1 << 63 } else { 0 },
        4,
        true,
        Some(super::fault_law::Retained::Scalar(law)),
    )
    .is_ok()
}
#[test]
fn scalar_words_require_actual_operation_and_recovery_permission() {
    use PcuRangePolicy::{Clamp, Reject};
    let add = PcuCheckedScalarFaultLaw::integer_binary(
        PcuScalarType::U512,
        PcuDispatchIntegerBinaryOp::Add,
        Reject,
    )
    .unwrap();
    assert!(valid(3, false, add));
    for tag in [1, 2, 4, 5] {
        assert!(!valid(tag, false, add));
    }
    assert!(!valid(3, true, add));
    let clamp = PcuCheckedScalarFaultLaw::integer_binary(
        PcuScalarType::U512,
        PcuDispatchIntegerBinaryOp::Add,
        Clamp,
    )
    .unwrap();
    assert!(valid(3, true, clamp));
    assert!(!valid(3, false, clamp));
    for scalar in [PcuScalarType::I128, PcuScalarType::U128] {
        let law = PcuCheckedScalarFaultLaw::integer_div_rem(scalar).unwrap();
        assert!(valid(1, false, law));
        assert_eq!(valid(2, false, law), scalar == PcuScalarType::I128);
        for tag in [3, 4, 5] {
            assert!(!valid(tag, false, law));
        }
    }
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        let add = PcuCheckedScalarFaultLaw::float_binary(
            PcuScalarType::F16,
            PcuDispatchFloatBinaryOp::Add,
            Reject,
            policy,
        )
        .unwrap();
        assert!(!valid(1, false, add));
        assert!(valid(5, false, add));
        assert_eq!(
            valid(4, false, add),
            policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
        );
        let unary = PcuCheckedScalarFaultLaw::float_unary(
            PcuScalarType::F64,
            PcuDispatchFloatUnaryOp::Relu,
            Clamp,
            policy,
        )
        .unwrap();
        assert!(!valid(3, true, unary));
        assert_eq!(
            valid(4, true, unary),
            policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
        );
    }
    let widen = PcuCheckedScalarFaultLaw::float_conversion(
        PcuDispatchCheckedFloatConversion::F32ToF64,
        Clamp,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    );
    assert!(valid(5, false, widen));
    assert!(!valid(3, true, widen));
    assert!(!valid(4, true, widen));
}
#[test]
fn scalar_sentinel_and_extent_are_independent_of_fault_permissions() {
    let law = PcuCheckedScalarFaultLaw::integer_div_rem(PcuScalarType::U64).unwrap();
    assert_eq!(
        super::decode_fault_word_under_law(
            u64::MAX,
            0,
            true,
            Some(super::fault_law::Retained::Scalar(law))
        )
        .unwrap(),
        None
    );
    assert!(
        super::decode_fault_word_under_law(
            (3 << 3) | 1,
            4,
            true,
            Some(super::fault_law::Retained::Scalar(law))
        )
        .is_ok()
    );
    assert!(
        super::decode_fault_word_under_law(
            (4 << 3) | 1,
            4,
            true,
            Some(super::fault_law::Retained::Scalar(law))
        )
        .is_err()
    );
}
#[test]
fn private_compound_event_extent_is_not_a_scalar_launch_or_output_extent() {
    // Existing compound factories use MSE3*N+1, SGD2*N, MatMul2*cells*K.
    for (event, extent) in [
        (9, 10),
        (7, 8),
        (23, 24),
        (u64::from(u32::MAX) + 7, u64::from(u32::MAX) + 8),
    ] {
        let word = (event << 3) | 3;
        assert!(super::decode_fault_word_under_law(word, extent, false, None).is_ok());
        assert!(super::decode_fault_word_under_law(word, event, false, None).is_err());
    }
    let above_scalar = ((u64::from(u32::MAX) + 1) << 3) | 3;
    assert!(super::decode_fault_word(above_scalar).is_err());
}
#[test]
fn genuine_direct_and_grid_joint_sources_capture_signedness_without_warm_ir() {
    let direct = source::repeated_bindings::<i128>();
    source::__repeated_ir_with_float_underflow_policy::<i128, 65>(
        &direct,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
        PcuImplementationRequirements::DEFAULT,
    )
    .unwrap()
    .with_ir(|ir| {
        assert_eq!(
            super::checked_scalar_fault_law(ir),
            PcuCheckedScalarFaultLaw::integer_div_rem(PcuScalarType::I128)
        );
        assert_eq!(super::checked_fault_extent(ir), 65);
    });
    let grid = source::grid_zero_bindings::<fusion_pcu::PcuU512>();
    source::__grid_zero_ir_with_float_underflow_policy::<fusion_pcu::PcuU512, 65>(
        &grid,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuRangePolicy::Reject,
        PcuImplementationRequirements::DEFAULT,
    )
    .unwrap()
    .with_ir(|ir| {
        assert_eq!(
            super::checked_scalar_fault_law(ir),
            PcuCheckedScalarFaultLaw::integer_div_rem(PcuScalarType::U512)
        );
        assert_eq!(super::checked_fault_extent(ir), 65);
        assert_eq!(ir.entry.logical_shape[0], 17);
    });
}

#[cfg(feature = "tensor")]
#[test]
fn compound_words_keep_exact_step_masks_before_publication() {
    use fusion_pcu::dialect::tensor::TensorStrictFaultDomain;
    let policy = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    for (domain, event, tag) in [
        (
            TensorStrictFaultDomain::matmul(PcuScalarType::F32, 1, 3, policy).unwrap(),
            4,
            3,
        ),
        (
            TensorStrictFaultDomain::sgd(PcuScalarType::F64, 3, policy).unwrap(),
            5,
            5,
        ),
        (
            TensorStrictFaultDomain::mse(PcuScalarType::F32, 3, policy).unwrap(),
            9,
            4,
        ),
    ] {
        let law = Some(super::fault_law::Retained::Compound(domain));
        assert!(
            super::decode_fault_word_under_law(
                (event << 3) | tag,
                domain.event_extent(),
                false,
                law
            )
            .is_ok()
        );
        assert!(
            super::decode_fault_word_under_law((event << 3) | 2, domain.event_extent(), false, law)
                .is_err()
        );
        assert!(
            super::decode_fault_word_under_law(
                (1 << 63) | (event << 3) | 3,
                domain.event_extent(),
                false,
                law
            )
            .is_err()
        );
    }
    let domain = TensorStrictFaultDomain::mse(PcuScalarType::F32, 3, policy).unwrap();
    let law = Some(super::fault_law::Retained::Compound(domain));
    for tag in [1, 3, 5] {
        assert!(
            super::decode_fault_word_under_law((9 << 3) | tag, domain.event_extent(), false, law)
                .is_err()
        );
    }
    // Same-format subtraction cannot be IEEE tiny-inexact; final division can.
    assert!(super::decode_fault_word_under_law(4, domain.event_extent(), false, law).is_err());
}

#[allow(dead_code)] // Genuine composition is executed through generated IR in this cold regression.
mod composed_source {
    #[rustfmt::skip]
    use pcu_facade::{
        pcu,
        PcuCheckedFloat,
    };
    #[pcu(crate_path=::pcu_facade, invocations=4)]
    pub fn composed<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
        let id = pcu::context::global_invocation_id();
        output[id] = (left[id] + right[id]) * left[id];
    }
}
#[test]
fn previously_admitted_composed_float_map_retains_fault_domain() {
    fn format<T: pcu_facade::PcuCheckedFloat>() {
        let bindings = composed_source::composed_bindings::<T>();
        let builder = composed_source::composed_ir::<T>(&bindings).unwrap();
        builder.with_ir(|ir| {
            crate::lower_dispatch_to_hip_source(ir).unwrap();
            let law = super::checked_scalar_fault_law(ir).expect(
                "previously admitted composed checked floating maps retain a cold fault law",
            );
            assert!(valid(3, false, law));
            assert!(valid(4, false, law)); // IEEE multiply may produce tiny-inexact underflow.
            assert!(valid(5, false, law));
            assert!(!valid(1, false, law)); // Neither Add nor Mul divides.
            assert!(!valid(3, true, law)); // Reject never publishes a recovered notice.
            let mut operations = ir.ops.to_vec();
            for operation in &mut operations {
                if let fusion_pcu::PcuDispatchOp::Data(
                    fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary {
                        underflow_policy, ..
                    },
                ) = operation
                {
                    *underflow_policy = PcuFloatUnderflowPolicy::AllowGradualUnderflow;
                }
            }
            let scoped = fusion_pcu::PcuDispatchKernelIr {
                ops: &operations,
                ..*ir
            };
            let scoped_law = super::checked_scalar_fault_law(&scoped).unwrap();
            assert!(!valid(4, false, scoped_law));
            let range = operations
                .iter_mut()
                .find_map(|operation| match operation {
                    fusion_pcu::PcuDispatchOp::Data(
                        fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary { range_policy, .. },
                    ) => Some(range_policy),
                    _ => None,
                })
                .unwrap();
            *range = PcuRangePolicy::Clamp;
            let mixed_range = fusion_pcu::PcuDispatchKernelIr {
                ops: &operations,
                ..*ir
            };
            assert!(super::checked_scalar_fault_law(&mixed_range).is_none());
            for operation in &mut operations {
                if let fusion_pcu::PcuDispatchOp::Data(
                    fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary { range_policy, .. },
                ) = operation
                {
                    *range_policy = PcuRangePolicy::Clamp;
                }
            }
            let clamp = fusion_pcu::PcuDispatchKernelIr {
                numerical_requirements: PcuImplementationRequirements {
                    range_policy: PcuRangePolicy::Clamp,
                    ..ir.numerical_requirements
                },
                ops: &operations,
                ..*ir
            };
            let clamp_law = super::checked_scalar_fault_law(&clamp).unwrap();
            assert!(valid(3, true, clamp_law));
            assert!(!valid(3, false, clamp_law));
            assert!(valid(5, false, clamp_law));
            assert!(!valid(5, true, clamp_law));
        });
    }
    format::<f32>();
    format::<f64>();
}

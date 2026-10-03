//! Exact cold resource domains of the public synchronous adapter; no device required.
use crate::owned_dispatch::scalar_fault_law_tests::source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuRangePolicy,
};
#[test]
fn legacy_grid_projects_unread_and_retains_full_logical_spans() {
    let bindings = source::grid_zero_bindings::<i64>();
    let builder = source::__grid_zero_ir_with_float_underflow_policy::<i64, 65>(
        &bindings,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
        PcuImplementationRequirements::DEFAULT,
    )
    .unwrap();
    builder.with_ir(|ir| {
        let requirements = super::binding_requirements(ir).unwrap();
        assert_eq!(
            requirements.iter().map(|r| r.target).collect::<Vec<_>>(),
            [
                PcuBindingRef::new(0, 1),
                PcuBindingRef::new(0, 2),
                PcuBindingRef::new(0, 3)
            ]
        );
        assert!(requirements.iter().all(|r| r.min_required_bytes == 65 * 8));
        assert_eq!(crate::owned_dispatch::checked_fault_extent(ir), 65);
        assert_eq!(ir.entry.logical_shape[0], 17);
    });
}
#[test]
fn legacy_only_zero_read_requires_one_element_while_joint_outputs_require_full_extent() {
    let bindings = source::repeated_bindings::<fusion_pcu::PcuU512>();
    let builder = source::__repeated_ir_with_float_underflow_policy::<fusion_pcu::PcuU512, 65>(
        &bindings,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
        PcuImplementationRequirements::DEFAULT,
    )
    .unwrap();
    builder.with_ir(|ir| {
        let mut ops = ir.ops.to_vec();
        for op in &mut ops {
            if let fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::BindingLoad {
                index,
                ..
            }) = op
            {
                *index = fusion_pcu::PcuDispatchIndex::BindingElementZero;
            }
        }
        let changed = fusion_pcu::PcuDispatchKernelIr { ops: &ops, ..*ir };
        let requirements = super::binding_requirements(&changed).unwrap();
        assert_eq!(
            requirements
                .iter()
                .map(|r| r.min_required_bytes)
                .collect::<Vec<_>>(),
            [65 * 64, 65 * 64, 64]
        );
    });
}
#[test]
fn legacy_fault_domain_rejects_out_of_extent_even_for_recovered_known_tag() {
    for recovered in [0, 1u64 << 63] {
        assert!(super::decode_fault_word_in_extent(recovered | (8 << 3) | 3, 4).is_err());
        assert!(super::decode_fault_word_in_extent(recovered | (8 << 3) | 3, 19).is_ok());
    }
}

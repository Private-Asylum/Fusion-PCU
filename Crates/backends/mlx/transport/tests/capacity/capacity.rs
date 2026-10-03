//! Full native capacities are a separate cold contract from initial read minima.
#[rustfmt::skip]
use crate::{
    MlxError,
    MlxTransportPlan,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuScalarType,
};
use super::graph;

#[test]
fn all_twenty_two_full_capacities_keep_actual_initial_read_minima() {
    for scalar in PcuScalarType::ALL {
        if scalar.bit_width() < 8 {
            continue;
        }
        for grid in [false, true] {
            graph::visit(
                scalar,
                17,
                grid,
                PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                |ir| {
                    let plan = MlxTransportPlan::assess(ir, scalar).unwrap();
                    assert_eq!(plan.input_element_counts(), [17, 1, 0, 0]);
                    assert_eq!(plan.assess_input_extents(&[24, 4]), Ok([24, 4, 0, 0]));
                    assert_eq!(plan.input_element_counts(), [17, 1, 0, 0]);
                    for invalid in [&[24][..], &[16, 4], &[24, 0], &[24, usize::MAX]] {
                        assert_eq!(
                            plan.assess_input_extents(invalid),
                            Err(MlxError::InvalidExtent)
                        );
                    }
                },
            );
            graph::visit_four(
                scalar,
                17,
                grid,
                PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                |ir| {
                    let plan = MlxTransportPlan::assess(ir, scalar).unwrap();
                    assert_eq!(
                        plan.input_bindings(),
                        [graph::INPUT, graph::SEED, graph::STAGE, graph::OUTPUT]
                    );
                    assert_eq!(plan.input_element_counts(), [17, 1, 17, 17]);
                    assert_eq!(
                        plan.assess_input_extents(&[24, 4, 20, 22]),
                        Ok([24, 4, 20, 22])
                    );
                    assert_eq!(plan.outputs().len(), 2);
                    assert!(plan.native_source().contains("auto v2=input2[i];"));
                    assert!(plan.native_source().contains("auto v3=input3[i];"));
                    assert!(plan.native_source().contains("output1[i]=v2;"));
                },
            );
        }
    }
}

#[path = "native/native.rs"]
mod native;

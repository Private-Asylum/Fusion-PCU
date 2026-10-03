//! Pure exact-role/SSA tests; native qualification remains a separate endpoint gate.
#[path = "graph/graph.rs"]
mod graph;
use super::MlxTransportPlan;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuScalarType,
    PcuReproducibility,
};
#[test]
fn all_twenty_two_saved_ssa_keep_original_roles_and_skip_stage_initial_upload() {
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
                    assert_eq!(plan.element_count(), 17);
                    assert_eq!(plan.requirements(), ir.numerical_requirements);
                    assert_eq!(plan.resources().len(), 4);
                    assert_eq!(plan.input_bindings(), [graph::INPUT, graph::SEED]);
                    assert_eq!(plan.declared_bindings().len(), 5);
                    assert_eq!(
                        plan.inputs()
                            .iter()
                            .map(|r| (r.binding, r.minimum_initial_read_elements))
                            .collect::<Vec<_>>(),
                        [(graph::INPUT, 17), (graph::SEED, 1)]
                    );
                    assert_eq!(
                        plan.outputs().iter().map(|r| r.binding).collect::<Vec<_>>(),
                        [graph::STAGE, graph::OUTPUT]
                    );
                    let source = plan.native_source();
                    assert!(source.contains("auto v1=v0;"));
                    assert!(source.contains("auto v2=input1[limb];"));
                    assert!(source.contains("output0[i]=v2;"));
                    assert!(source.contains("output1[i]=v1;"));
                    assert!(!source.contains("float"));
                },
            );
        }
    }
}
#[test]
fn packed_and_portable_refuse_without_creating_native_arrays() {
    for scalar in [PcuScalarType::Bool, PcuScalarType::I4, PcuScalarType::U4] {
        graph::visit(
            scalar,
            17,
            false,
            PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            |ir| assert!(MlxTransportPlan::assess(ir, scalar).is_err()),
        );
    }
    let mut request = PcuDispatchKernelIr::DEFAULT_REQUIREMENTS;
    request.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    graph::visit(PcuScalarType::F32, 17, false, request, |ir| {
        assert!(MlxTransportPlan::assess(ir, PcuScalarType::F32).is_err());
    });
}

#[path = "native/native.rs"]
mod native;

#[path = "mixed/mixed.rs"]
mod mixed;

#[path = "aggregate/aggregate.rs"]
mod aggregate;

#[path = "capacity/capacity.rs"]
mod capacity;

#[cfg(feature = "benchmark-control")]
#[path = "control/control.rs"]
mod control;

//! Exact schema variants and native terminal host publication.
use super::*;
use fusion_pcu::{PcuFloatUnderflowPolicy as Policy, PcuRangePolicy as Range, PcuReproducibility};
#[path = "graph/graph.rs"]
mod graph;
#[test]
fn conversion_schema_retains_all_canonical_layouts_and_original_requirements() {
    for conversion in [
        PcuDispatchCheckedFloatConversion::F32ToF64,
        PcuDispatchCheckedFloatConversion::F64ToF32,
    ] {
        for range in [Range::Reject, Range::Clamp] {
            for policy in [
                Policy::IeeeAfterRounding,
                Policy::RejectSubnormalResult,
                Policy::AllowGradualUnderflow,
            ] {
                for broadcast in [false, true] {
                    for grid in [false, true] {
                        graph::with(conversion, 65, range, policy, broadcast, grid, |kernel| {
                            let plan = MlxCheckedConversionPlan::assess(kernel).unwrap();
                            assert_eq!(plan.requirements(), kernel.numerical_requirements);
                            assert_eq!(plan.count, 65);
                            assert_eq!(plan.input, graph::INPUT);
                            assert_eq!(plan.output, graph::OUTPUT);
                            assert_eq!(plan.broadcast, broadcast);
                            assert_eq!(plan.conversion, conversion);
                            let mut portable = *kernel;
                            portable
                                .numerical_requirements
                                .numerical_options
                                .reproducibility = PcuReproducibility::PortableV1;
                            assert!(MlxCheckedConversionPlan::assess(&portable).is_err());
                            let mut wrong_policy = *kernel;
                            wrong_policy.numerical_requirements.range_policy =
                                if range == Range::Reject {
                                    Range::Clamp
                                } else {
                                    Range::Reject
                                };
                            assert!(MlxCheckedConversionPlan::assess(&wrong_policy).is_err());
                        });
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual official MLX0.32.3 GPU runtime."]
fn conversion_host_private_fatal_rollback_recovered_publication_retry_and_tails() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let backend = MlxConversionHostBackend::new(session);
    for range in [Range::Reject, Range::Clamp] {
        for grid in [false, true] {
            graph::with(
                PcuDispatchCheckedFloatConversion::F64ToF32,
                3,
                range,
                Policy::IeeeAfterRounding,
                false,
                grid,
                |kernel| {
                    let mut prepared = backend.prepare_host_kernel(kernel).unwrap();
                    let mut output = [91.0_f32; 5];
                    let invalid = [1.0, f64::INFINITY, 2.0];
                    let mut args = [
                        PcuHostArgument::read_write(graph::OUTPUT, &mut output),
                        PcuHostArgument::read(graph::INPUT, &invalid),
                    ];
                    assert!(
                        matches!(prepared.call(&mut args),Err(MlxError::Arithmetic(fault)) if !fault.recovered)
                    );
                    assert_eq!(output.map(f32::to_bits), [91.0_f32; 5].map(f32::to_bits));
                    let recovered = [1.0, f64::MAX, 2.0];
                    let mut args = [
                        PcuHostArgument::read(graph::INPUT, &recovered),
                        PcuHostArgument::read_write(graph::OUTPUT, &mut output),
                    ];
                    assert!(
                        matches!(prepared.call(&mut args),Err(MlxError::Arithmetic(fault)) if fault.recovered==(range==Range::Clamp))
                    );
                    assert_eq!(
                        output.map(f32::to_bits),
                        (if range == Range::Clamp {
                            [1.0, f32::MAX, 2.0, 91.0, 91.0]
                        } else {
                            [91.0; 5]
                        })
                        .map(f32::to_bits)
                    );
                    let healthy = [1.0, 3.0, -2.0];
                    let mut args = [
                        PcuHostArgument::read(graph::INPUT, &healthy),
                        PcuHostArgument::read_write(graph::OUTPUT, &mut output),
                    ];
                    prepared.call(&mut args).unwrap();
                    assert_eq!(
                        output.map(f32::to_bits),
                        [1.0_f32, 3.0, -2.0, 91.0, 91.0].map(f32::to_bits)
                    );
                    let short = [1.0, 2.0];
                    let mut args = [
                        PcuHostArgument::read(graph::INPUT, &short),
                        PcuHostArgument::read_write(graph::OUTPUT, &mut output),
                    ];
                    assert_eq!(prepared.call(&mut args), Err(MlxError::InvalidExtent));
                    assert_eq!(
                        output.map(f32::to_bits),
                        [1.0_f32, 3.0, -2.0, 91.0, 91.0].map(f32::to_bits)
                    );
                },
            );
        }
    }
}

#[path = "aggregate/aggregate.rs"]
mod aggregate;

#[path = "offers/offers.rs"]
mod offers;

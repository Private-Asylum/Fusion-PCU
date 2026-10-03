//! Genuine static session factory preserves unique snapshots and plural writer metadata.
#[rustfmt::skip]
use crate::{
    MlxRuntime,
    MlxPreparedDispatchKernel,
    MlxDispatchOutputLayout,
    MlxDispatchCompletion,
    MlxBinaryInput,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuHostKernelBackend,
    PcuScalarType,
    PcuDispatchKernelIr,
};
use super::graph;
#[test]
#[ignore = "requires genuine MLX transport session aggregate and explicit GPU completion"]
fn all_twenty_two_static_transport_arity_and_actual_snapshot_order() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
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
                    let mut prepared = session.prepare_host_kernel(ir).unwrap();
                    assert!(matches!(prepared, MlxPreparedDispatchKernel::Transport(_)));
                    assert_eq!(prepared.input_bindings(), [graph::INPUT, graph::SEED]);
                    assert_eq!(prepared.scalar_type(), scalar);
                    let width = usize::from(scalar.bit_width()) / 8;
                    assert_eq!(
                        prepared.output_layout(),
                        MlxDispatchOutputLayout::Transport {
                            outputs: [
                                Some((graph::STAGE, 17 * width)),
                                Some((graph::OUTPUT, 17 * width))
                            ],
                        }
                    );
                    let input: Vec<u8> = (0..17 * width)
                        .map(|i| u8::try_from(i % 256).unwrap().wrapping_mul(31))
                        .collect();
                    let seed = vec![0xff; width];
                    let inputs = [
                        MlxBinaryInput::HostBytes {
                            target: graph::SEED,
                            scalar,
                            bytes: &seed,
                        },
                        MlxBinaryInput::HostBytes {
                            target: graph::INPUT,
                            scalar,
                            bytes: &input,
                        },
                    ];
                    let MlxDispatchCompletion::Transport(completed) =
                        prepared.execute_inputs(&inputs).unwrap()
                    else {
                        panic!("transport output cardinality");
                    };
                    assert_eq!(completed.output_count(), 2);
                    let [Some(stage), Some(output)] = completed.into_outputs() else {
                        panic!("both writers");
                    };
                    let mut actual = vec![0; 17 * width];
                    stage.read_bytes_into(&mut actual).unwrap();
                    assert_eq!(actual, seed.repeat(17));
                    output.read_bytes_into(&mut actual).unwrap();
                    assert_eq!(actual, input);
                    assert!(!prepared.last_call_may_have_written());
                    assert!(!prepared.last_call_completion_uncertain());
                    stage.release().unwrap();
                    output.release().unwrap();
                },
            );
        }
    }
}

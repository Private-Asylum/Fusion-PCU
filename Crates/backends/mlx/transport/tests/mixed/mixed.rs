//! Actual initial input directions with private immutable sibling completion.
#[rustfmt::skip]
use crate::{
    MlxRuntime,
    MlxError,
    MlxTransportInput,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuHostKernelBackend,
    PcuHostDispatchError,
    PcuScalarType,
};
use super::graph;
fn raw(scalar: PcuScalarType, count: usize, phase: u8) -> Vec<u8> {
    let width = usize::from(scalar.bit_width()) / 8;
    (0..count * width)
        .map(|index| {
            if index / width % 7 == 1 {
                0xff
            } else {
                u8::try_from(index % 256)
                    .unwrap()
                    .wrapping_mul(29)
                    .wrapping_add(phase)
            }
        })
        .collect()
}
#[test]
#[ignore = "requires genuine MLX GPU transport; record external activity, correctness only"]
fn all_twenty_two_actual_initial_host_resident_directions_and_private_siblings() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
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
                    let backend = session.transport_host_backend();
                    let mut prepared = backend.prepare_host_kernel(ir).unwrap();
                    for phase in [0, 31, 79] {
                        let width = usize::from(scalar.bit_width()) / 8;
                        let host = raw(scalar, 24, phase);
                        let seed = raw(scalar, 1, phase.wrapping_add(13));
                        let input = session
                            .upload_transport_bytes(scalar, 17, &host[..17 * width])
                            .unwrap();
                        let seed_owner = session.upload_transport_bytes(scalar, 1, &seed).unwrap();
                        let foreign_input = foreign
                            .upload_transport_bytes(scalar, 17, &host[..17 * width])
                            .unwrap();
                        let initial = MlxTransportInput::Resident {
                            target: graph::INPUT,
                            array: &input,
                        };
                        let seed_input = MlxTransportInput::Resident {
                            target: graph::SEED,
                            array: &seed_owner,
                        };
                        let conflict = MlxTransportInput::Resident {
                            target: graph::INPUT,
                            array: &foreign_input,
                        };
                        assert_eq!(
                            prepared.execute_inputs(&[conflict, seed_input]).err(),
                            Some(PcuHostDispatchError::Backend(MlxError::ForeignSession))
                        );
                        assert_eq!(
                            prepared.execute_inputs(&[initial, initial]).err(),
                            Some(PcuHostDispatchError::Duplicate(graph::INPUT))
                        );
                        assert_eq!(
                            prepared.execute_inputs(&[initial]).err(),
                            Some(PcuHostDispatchError::Missing(graph::SEED))
                        );
                        let short = MlxTransportInput::HostBytes {
                            target: graph::SEED,
                            scalar,
                            bytes: &[],
                        };
                        assert_eq!(
                            prepared.execute_inputs(&[initial, short]).err(),
                            Some(PcuHostDispatchError::BufferTooSmall(graph::SEED))
                        );
                        directions(&mut prepared, scalar, &host, &seed, &input, &seed_owner);
                        let mut original = vec![0; 17 * width];
                        input.read_bytes_into(&mut original).unwrap();
                        assert_eq!(original, &host[..17 * width]);
                        input.release().unwrap();
                        seed_owner.release().unwrap();
                        foreign_input.release().unwrap();
                    }
                },
            );
        }
    }
}

fn directions(
    prepared: &mut crate::MlxPreparedTransportHostKernel,
    scalar: PcuScalarType,
    host: &[u8],
    seed: &[u8],
    input: &crate::MlxEncodedArray,
    seed_owner: &crate::MlxEncodedArray,
) {
    let width = usize::from(scalar.bit_width()) / 8;
    let initial = MlxTransportInput::Resident {
        target: graph::INPUT,
        array: input,
    };
    let seed_input = MlxTransportInput::Resident {
        target: graph::SEED,
        array: seed_owner,
    };
    for mask in 0..4 {
        let input = if mask & 1 == 0 {
            MlxTransportInput::HostBytes {
                target: graph::INPUT,
                scalar,
                bytes: host,
            }
        } else {
            initial
        };
        let seed_input = if mask & 2 == 0 {
            MlxTransportInput::HostBytes {
                target: graph::SEED,
                scalar,
                bytes: seed,
            }
        } else {
            seed_input
        };
        // Deliberately permuted role order; no function-parameter ordinal assumption.
        let completed = prepared.execute_inputs(&[seed_input, input]).unwrap();
        assert!(!prepared.last_call_may_have_written());
        assert!(!prepared.last_call_completion_uncertain());
        let [Some(stage), Some(output)] = completed.into_outputs() else {
            panic!("two fresh writers");
        };
        let mut stage_bytes = vec![0x91; 19 * width];
        let mut output_bytes = vec![0x73; 20 * width];
        stage
            .read_bytes_into(&mut stage_bytes[..17 * width])
            .unwrap();
        output
            .read_bytes_into(&mut output_bytes[..17 * width])
            .unwrap();
        assert_eq!(&stage_bytes[..17 * width], seed.repeat(17));
        assert_eq!(&output_bytes[..17 * width], &host[..17 * width]);
        assert_eq!(&stage_bytes[17 * width..], vec![0x91; 2 * width]);
        assert_eq!(&output_bytes[17 * width..], vec![0x73; 3 * width]);
        stage.release().unwrap();
        output.release().unwrap();
    }
}

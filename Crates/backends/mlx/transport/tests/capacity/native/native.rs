//! Real full-size immutable inputs, logical-prefix results and exact prepared shapes.
#[rustfmt::skip]
use crate::{
    MlxEncodedArray,
    MlxError,
    MlxPreparedTransportHostKernel,
    MlxRuntime,
    MlxSession,
    MlxTransportInput,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuScalarType,
};
use super::super::graph;
const MINIMUM: [usize; 4] = [17, 1, 17, 17];
const FULL: [usize; 4] = [24, 4, 20, 22];
fn raw(scalar: PcuScalarType, count: usize, phase: u8) -> Vec<u8> {
    let width = usize::from(scalar.bit_width()) / 8;
    (0..count * width)
        .map(|index| match (index / width) % 7 {
            1 => 0xff,
            2 => {
                if index % width == width - 1 {
                    0x80
                } else {
                    0
                }
            }
            _ => u8::try_from(index % 256)
                .unwrap()
                .wrapping_mul(37)
                .wrapping_add(phase),
        })
        .collect()
}
#[test]
#[ignore = "requires genuine MLX GPU full-capacity transport; correctness only"]
fn all_twenty_two_full_capacities_and_four_actual_snapshot_directions() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    for scalar in PcuScalarType::ALL {
        if scalar.bit_width() < 8 {
            continue;
        }
        for grid in [false, true] {
            for four in [false, true] {
                let visit =
                    |ir: &PcuDispatchKernelIr<'_>| qualify(&session, &foreign, scalar, ir, four);
                if four {
                    graph::visit_four(
                        scalar,
                        17,
                        grid,
                        PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                        visit,
                    );
                } else {
                    graph::visit(
                        scalar,
                        17,
                        grid,
                        PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                        visit,
                    );
                }
            }
        }
    }
    eprintln!(
        "Capacity witness: 2640 changing successful calls, 880 distinct cold shape/direction plans; default original normal tuple only. Full synthetic cold input bytes=width*sum(actual capacities), 18..28 logical elements for two snapshots and 52..70 for four. Native SDK heap/JIT internals unobserved; no warm census or latency claim."
    );
}
fn qualify(
    session: &MlxSession,
    foreign: &MlxSession,
    scalar: PcuScalarType,
    ir: &PcuDispatchKernelIr<'_>,
    four: bool,
) {
    let arity = if four { 4 } else { 2 };
    let backend = session.transport_host_backend();
    let mut exact = backend.prepare_host_kernel(ir).unwrap();
    assert_eq!(exact.prepared_input_element_counts(), &MINIMUM[..arity]);
    for mask in 0..1 << arity {
        let extents = std::array::from_fn::<_, 4, _>(|slot| {
            if mask & (1 << slot) == 0 {
                MINIMUM[slot]
            } else {
                FULL[slot]
            }
        });
        let mut prepared = backend
            .prepare_host_kernel_with_input_extents(ir, &extents[..arity])
            .unwrap();
        assert_eq!(prepared.prepared_input_element_counts(), &extents[..arity]);
        for phase in [0_u8, 31, 79] {
            let bytes: [Vec<u8>; 4] = std::array::from_fn(|slot| {
                raw(
                    scalar,
                    FULL[slot],
                    phase.wrapping_add(u8::try_from(slot).unwrap() * 13),
                )
            });
            let owners: [Option<MlxEncodedArray>; 4] = std::array::from_fn(|slot| {
                (slot < arity).then(|| {
                    session
                        .upload_transport_bytes(scalar, FULL[slot], &bytes[slot])
                        .unwrap()
                })
            });
            let inputs: Vec<_> = (0..arity)
                .map(|slot| {
                    input(
                        &prepared,
                        slot,
                        mask,
                        scalar,
                        &bytes[slot],
                        owners[slot].as_ref().unwrap(),
                    )
                })
                .collect();
            let oversized: Vec<_> = (0..arity)
                .map(|slot| MlxTransportInput::Resident {
                    target: exact.plan().input_bindings()[slot],
                    array: owners[slot].as_ref().unwrap(),
                })
                .collect();
            assert_eq!(
                exact.execute_inputs(&oversized).err(),
                Some(PcuHostDispatchError::Backend(MlxError::InvalidExtent))
            );
            guards(
                &mut prepared,
                foreign,
                scalar,
                &inputs,
                &bytes[0],
                &owners,
                mask,
            );
            let reversed: Vec<_> = inputs.iter().rev().copied().collect();
            let outputs = prepared.execute_inputs(&reversed).unwrap().into_outputs();
            verify(outputs, scalar, &bytes, four);
            assert!(!prepared.last_call_may_have_written());
            assert!(!prepared.last_call_completion_uncertain());
            for (slot, owner) in owners.into_iter().enumerate() {
                if let Some(owner) = owner {
                    let mut unchanged = vec![0; bytes[slot].len()];
                    owner.read_bytes_into(&mut unchanged).unwrap();
                    assert_eq!(unchanged, bytes[slot]);
                    owner.release().unwrap();
                }
            }
        }
    }
}
fn input<'a>(
    prepared: &MlxPreparedTransportHostKernel,
    slot: usize,
    mask: usize,
    scalar: PcuScalarType,
    bytes: &'a [u8],
    owner: &'a MlxEncodedArray,
) -> MlxTransportInput<'a> {
    let target = prepared.plan().input_bindings()[slot];
    if mask & (1 << slot) == 0 {
        MlxTransportInput::HostBytes {
            target,
            scalar,
            bytes,
        }
    } else {
        MlxTransportInput::Resident {
            target,
            array: owner,
        }
    }
}
fn guards(
    prepared: &mut MlxPreparedTransportHostKernel,
    foreign: &MlxSession,
    scalar: PcuScalarType,
    inputs: &[MlxTransportInput<'_>],
    first_bytes: &[u8],
    owners: &[Option<MlxEncodedArray>; 4],
    mask: usize,
) {
    let first = prepared.plan().input_bindings()[0];
    let mut bad = inputs.to_vec();
    if mask & 1 == 0 {
        bad[0] = MlxTransportInput::Resident {
            target: first,
            array: owners[0].as_ref().unwrap(),
        };
        assert_eq!(
            prepared.execute_inputs(&bad).err(),
            Some(PcuHostDispatchError::Backend(MlxError::InvalidExtent))
        );
    } else {
        bad[0] = MlxTransportInput::HostBytes {
            target: first,
            scalar,
            bytes: first_bytes,
        };
        assert_eq!(
            prepared.execute_inputs(&bad).err(),
            Some(PcuHostDispatchError::Backend(MlxError::InvalidExtent))
        );
        let alien = foreign
            .upload_transport_bytes(scalar, FULL[0], first_bytes)
            .unwrap();
        bad[0] = MlxTransportInput::Resident {
            target: first,
            array: &alien,
        };
        assert_eq!(
            prepared.execute_inputs(&bad).err(),
            Some(PcuHostDispatchError::Backend(MlxError::ForeignSession))
        );
        alien.release().unwrap();
    }
}
fn verify(
    outputs: [Option<MlxEncodedArray>; 2],
    scalar: PcuScalarType,
    bytes: &[Vec<u8>; 4],
    four: bool,
) {
    let width = usize::from(scalar.bit_width()) / 8;
    for (slot, output) in outputs.into_iter().enumerate() {
        let output = output.unwrap();
        let mut actual = vec![0x97; 19 * width];
        output.read_bytes_into(&mut actual[..17 * width]).unwrap();
        let expected = if slot == 0 {
            bytes[1][..width].repeat(17)
        } else {
            bytes[if four { 2 } else { 0 }][..17 * width].to_vec()
        };
        assert_eq!(&actual[..17 * width], expected);
        assert_eq!(&actual[17 * width..], vec![0x97; 2 * width]);
        let mut short = vec![0x51; 17 * width - 1];
        assert!(output.read_bytes_into(&mut short).is_err());
        assert_eq!(short, vec![0x51; 17 * width - 1]);
        output.release().unwrap();
    }
}

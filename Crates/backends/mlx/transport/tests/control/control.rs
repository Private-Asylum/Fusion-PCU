//! Independent native source is checked against raw literal workload oracles.
#[rustfmt::skip]
use crate::{
    MlxError,
    MlxNativeTransportWorkload,
    MlxRuntime,
};
use fusion_pcu::PcuScalarType;
fn bytes(scalar: PcuScalarType, count: usize, phase: u8) -> Vec<u8> {
    let width = usize::from(scalar.bit_width()) / 8;
    (0..count * width)
        .map(|byte| match byte / width % 5 {
            0 => phase,
            1 => 0xff,
            2 => {
                if byte % width == width - 1 {
                    0x80
                } else {
                    0
                }
            }
            _ => u8::try_from(byte % 256)
                .unwrap()
                .wrapping_mul(37)
                .wrapping_add(phase),
        })
        .collect()
}
#[test]
#[ignore = "requires genuine MLX native control; independent codegen correctness only"]
fn all_twenty_two_independent_native_workloads_full_shapes_and_private_outputs() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    for scalar in PcuScalarType::ALL {
        if scalar.bit_width() < 8 {
            continue;
        }
        for workload in [
            MlxNativeTransportWorkload::SavedInput,
            MlxNativeTransportWorkload::PriorStage,
            MlxNativeTransportWorkload::SwapBanks,
        ] {
            let arity = workload.input_count();
            let counts = [24, 4, 20, 22];
            let prepared = session
                .prepare_native_transport_control(scalar, 17, workload, &counts[..arity])
                .unwrap();
            assert_eq!(prepared.prepared_input_element_counts(), &counts[..arity]);
            for phase in [0_u8, 31, 79] {
                let input: [Vec<u8>; 4] = std::array::from_fn(|slot| {
                    bytes(
                        scalar,
                        counts[slot],
                        phase.wrapping_add(u8::try_from(slot).unwrap() * 11),
                    )
                });
                let owners: Vec<_> = (0..arity)
                    .map(|slot| {
                        session
                            .upload_transport_bytes(scalar, counts[slot], &input[slot])
                            .unwrap()
                    })
                    .collect();
                let refs: Vec<_> = owners.iter().collect();
                assert_eq!(
                    prepared.execute(&refs[..arity - 1]).err(),
                    Some(MlxError::InvalidExtent)
                );
                let alien = foreign
                    .upload_transport_bytes(scalar, counts[0], &input[0])
                    .unwrap();
                let mut bad = refs.clone();
                bad[0] = &alien;
                assert_eq!(prepared.execute(&bad).err(), Some(MlxError::ForeignSession));
                alien.release().unwrap();
                let (stage, output) = prepared.execute(&refs).unwrap();
                let width = usize::from(scalar.bit_width()) / 8;
                let mut actual = vec![0x93; 19 * width];
                stage.read_bytes_into(&mut actual[..17 * width]).unwrap();
                let expected_stage = match workload {
                    MlxNativeTransportWorkload::SwapBanks => input[3][..17 * width].to_vec(),
                    _ => input[1][..width].repeat(17),
                };
                assert_eq!(&actual[..17 * width], expected_stage);
                assert_eq!(&actual[17 * width..], vec![0x93; 2 * width]);
                output.read_bytes_into(&mut actual[..17 * width]).unwrap();
                let saved = if matches!(workload, MlxNativeTransportWorkload::SavedInput) {
                    0
                } else {
                    2
                };
                assert_eq!(&actual[..17 * width], &input[saved][..17 * width]);
                assert_eq!(&actual[17 * width..], vec![0x93; 2 * width]);
                stage.release().unwrap();
                output.release().unwrap();
                for (slot, owner) in owners.into_iter().enumerate() {
                    let mut unchanged = vec![0; input[slot].len()];
                    owner.read_bytes_into(&mut unchanged).unwrap();
                    assert_eq!(unchanged, input[slot]);
                    owner.release().unwrap();
                }
            }
        }
    }
}

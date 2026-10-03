//! Exact byte oracle over genuine immutable MLX carriers and saved SSA outputs.
#[rustfmt::skip]
use crate::{
    MlxRuntime,
    MlxTransportPlan,
    MlxError,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuDispatchKernelIr,
};
use super::graph;
fn raw(scalar: PcuScalarType, count: usize, phase: u8) -> Vec<u8> {
    let width = usize::from(scalar.bit_width()) / 8;
    (0..count * width)
        .map(|index| match (index / width) % 7 {
            0 => phase,
            1 => 0xff,
            2 => {
                if index % width == width - 1 {
                    0x80
                } else {
                    0
                }
            }
            3 => u8::from(index.is_multiple_of(width)),
            _ => u8::try_from(index % 256)
                .unwrap()
                .wrapping_mul(37)
                .wrapping_add(phase),
        })
        .collect()
}
#[test]
#[ignore = "requires genuine pinned MLX GPU runtime; external activity recording, correctness only"]
fn all_twenty_two_saved_transport_actual_owners_terminal_tails_foreign_and_drop() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    for scalar in PcuScalarType::ALL {
        if scalar.bit_width() < 8 {
            continue;
        }
        let width = usize::from(scalar.bit_width()) / 8;
        for grid in [false, true] {
            graph::visit(
                scalar,
                17,
                grid,
                PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
                |ir| {
                    let plan = MlxTransportPlan::assess(ir, scalar).unwrap();
                    let kernel = session.prepare_transport_plan(plan).unwrap();
                    for phase in [0, 19, 73] {
                        let original = raw(scalar, 17, phase);
                        let seed = raw(scalar, 1, phase.wrapping_add(13));
                        let input = session
                            .upload_transport_bytes(scalar, 17, &original)
                            .unwrap();
                        let seed_owner = session.upload_transport_bytes(scalar, 1, &seed).unwrap();
                        let foreign_input = foreign
                            .upload_transport_bytes(scalar, 17, &original)
                            .unwrap();
                        assert_eq!(
                            kernel.execute(&[&foreign_input, &seed_owner]).err(),
                            Some(MlxError::ForeignSession)
                        );
                        assert!(kernel.execute(&[&seed_owner, &seed_owner]).is_err());
                        let completed = kernel.execute(&[&input, &seed_owner]).unwrap();
                        assert_eq!(completed.output_count(), 2);
                        let [Some(stage), Some(output)] = completed.into_outputs() else {
                            panic!("two exact writers");
                        };
                        let mut stage_bytes = vec![0x93; 19 * width];
                        let mut output_bytes = vec![0x79; 20 * width];
                        stage
                            .read_bytes_into(&mut stage_bytes[..17 * width])
                            .unwrap();
                        output
                            .read_bytes_into(&mut output_bytes[..17 * width])
                            .unwrap();
                        assert_eq!(&stage_bytes[..17 * width], seed.repeat(17));
                        assert_eq!(&output_bytes[..17 * width], original);
                        assert_eq!(&stage_bytes[17 * width..], vec![0x93; 2 * width]);
                        assert_eq!(&output_bytes[17 * width..], vec![0x79; 3 * width]);
                        let mut unchanged = vec![0; original.len()];
                        input.read_bytes_into(&mut unchanged).unwrap();
                        assert_eq!(unchanged, original);
                        let mut short = vec![0xb7; 17 * width - 1];
                        assert!(output.read_bytes_into(&mut short).is_err());
                        assert_eq!(short, vec![0xb7; 17 * width - 1]);
                        input.release().unwrap();
                        seed_owner.release().unwrap();
                        foreign_input.release().unwrap();
                        stage.release().unwrap();
                        output.release().unwrap();
                    }
                    let original = raw(scalar, 17, 87);
                    let seed = raw(scalar, 1, 34);
                    let input = session
                        .upload_transport_bytes(scalar, 17, &original)
                        .unwrap();
                    let seed_owner = session.upload_transport_bytes(scalar, 1, &seed).unwrap();
                    let completed = kernel.execute(&[&input, &seed_owner]).unwrap();
                    drop(kernel);
                    input.release().unwrap();
                    seed_owner.release().unwrap();
                    for (slot, owner) in completed.into_outputs().into_iter().flatten().enumerate()
                    {
                        let mut bytes = vec![0; 17 * width];
                        owner.read_bytes_into(&mut bytes).unwrap();
                        assert_eq!(
                            bytes,
                            if slot == 0 {
                                seed.repeat(17)
                            } else {
                                original.clone()
                            }
                        );
                        owner.release().unwrap();
                    }
                },
            );
        }
    }
}

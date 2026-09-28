//! Opt-in correctness fixture for the canonical owned `ROCm` execution path.

use std::error::Error;

use fusion_pcu::PcuExecutionNodeState;
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmOwnedDispatchBackend,
    RocmOwnedExecutionNode,
    RocmOwnedExecutionOperation,
    RocmOwnedExecutionTwoSlot,
    RocmTwoSlotExecutionStep,
};

const DEPENDENCIES: [&[usize]; 4] = [&[], &[0], &[1], &[2]];

/// Exercise a same-stream SGEMM chain, cross-stream device copy, and terminal readback.
#[allow(clippy::too_many_lines)]
pub fn run(backend: &RocmOwnedDispatchBackend) -> Result<(), Box<dyn Error>> {
    let left_values = [1.0_f32, 2.0, 3.0, 4.0];
    let identity_values = [1.0_f32, 0.0, 0.0, 1.0];
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>()
    };
    let expected = encode(&left_values);
    let identity_bytes = encode(&identity_values);
    let mut left = backend.allocate(expected.len())?;
    left.copy_from(&expected)?;
    let mut identity = backend.allocate(identity_bytes.len())?;
    identity.copy_from(&identity_bytes)?;
    let intermediate = backend.allocate(expected.len())?;
    let output = backend.allocate(expected.len())?;
    let copied = backend.allocate(expected.len())?;

    let sgemm_stream = backend.create_stream()?;
    let copy_stream = backend.create_stream()?;
    let readback_stream = backend.create_stream()?;
    let mut handle = backend.create_rocblas()?;
    handle.bind_stream(&sgemm_stream)?;

    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: DEPENDENCIES[0],
            operation: RocmOwnedExecutionOperation::Sgemm {
                handle: &handle,
                stream: &sgemm_stream,
                transpose_a: false,
                transpose_b: false,
                m: 2,
                n: 2,
                k: 2,
                alpha: 1.0,
                a: &left,
                lda: 2,
                b: &identity,
                ldb: 2,
                beta: 0.0,
                c: &intermediate,
                ldc: 2,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: DEPENDENCIES[1],
            operation: RocmOwnedExecutionOperation::Sgemm {
                handle: &handle,
                stream: &sgemm_stream,
                transpose_a: false,
                transpose_b: false,
                m: 2,
                n: 2,
                k: 2,
                alpha: 1.0,
                a: &intermediate,
                lda: 2,
                b: &identity,
                ldb: 2,
                beta: 0.0,
                c: &output,
                ldc: 2,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: DEPENDENCIES[2],
            operation: RocmOwnedExecutionOperation::DeviceCopy {
                stream: &copy_stream,
                source: &output,
                destination: &copied,
                bytes: expected.len(),
            },
        },
        RocmOwnedExecutionNode {
            dependencies: DEPENDENCIES[3],
            operation: RocmOwnedExecutionOperation::DeviceReadback {
                stream: &readback_stream,
                source: &copied,
                source_offset: 0,
                bytes: expected.len(),
            },
        },
    ];
    let mut scratch = [false; 4];
    let mut states = [PcuExecutionNodeState::Pending; 4];
    let mut execution =
        RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?;

    let mut submitted = Vec::new();
    let mut first_success_after = None;
    loop {
        match execution.step()? {
            RocmTwoSlotExecutionStep::Submitted { node } => submitted.push(node),
            RocmTwoSlotExecutionStep::Succeeded { .. } => {
                first_success_after.get_or_insert(submitted.len());
            }
            RocmTwoSlotExecutionStep::Complete => break,
            RocmTwoSlotExecutionStep::Waiting { node } => {
                return Err(
                    format!("correctness fixture unexpectedly waited at node {node}").into(),
                );
            }
            RocmTwoSlotExecutionStep::Failed { node, outcome } => {
                return Err(format!("correctness fixture node {node} failed: {outcome:?}").into());
            }
            RocmTwoSlotExecutionStep::Blocked { node } => {
                return Err(format!("correctness fixture node {node} was blocked").into());
            }
        }
    }
    if submitted.get(..2) != Some(&[0, 1]) || first_success_after.is_none_or(|count| count < 2) {
        return Err(format!(
            "same-stream SGEMMs did not both submit before the first wait (submitted {submitted:?}, first completion after {first_success_after:?})"
        )
        .into());
    }
    let actual = execution.take_readback(3)?;
    if actual.as_ref() != expected {
        return Err(
            format!("canonical dispatch output mismatch: {actual:?} != {expected:?}").into(),
        );
    }
    drop(execution);
    if states != [PcuExecutionNodeState::Succeeded; 4] {
        return Err(format!("canonical dispatch node states were {states:?}").into());
    }
    if !handle.is_usable() {
        return Err("rocBLAS handle remained busy after terminal readback".into());
    }
    println!(
        "Canonical ROCm dispatch correctness fixture passed (SGEMM → SGEMM → cross-stream copy → readback)"
    );
    Ok(())
}

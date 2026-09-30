//! Actual owned source composition retains native intermediates through one stream completion.
#![cfg(all(feature = "rocm", feature = "tensor"))]

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu(flag(non_strict), flag(native_compound))]
fn chain<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
    final_matrix: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let intermediate = pcu::matmul(left, right)?;
    pcu::matmul(&intermediate, final_matrix)
}

#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn actual_source_chain_changes_inputs_and_escapes_only_completed_output() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    })
    .unwrap();
    for phase in 1_i16..4 {
        let left = [[phase, phase + 1], [phase + 2, phase + 3]];
        let right = [[2_i16, 1], [0, 2]];
        let last = [[1_i16, 0], [1, 1]];
        // A*B*C: first column=3*a+2*b; second column=a+2*b.
        let expected = [
            3 * phase + 2 * (phase + 1),
            phase + 2 * (phase + 1),
            3 * (phase + 2) + 2 * (phase + 3),
            (phase + 2) + 2 * (phase + 3),
        ];
        let f32_output = chain::<f32>(
            &left.map(|row| row.map(f32::from)),
            &right.map(|row| row.map(f32::from)),
            &last.map(|row| row.map(f32::from)),
        )
        .unwrap();
        let f64_output = chain::<f64>(
            &left.map(|row| row.map(f64::from)),
            &right.map(|row| row.map(f64::from)),
            &last.map(|row| row.map(f64::from)),
        )
        .unwrap();
        global::clear_thread_cache().unwrap();
        let mut narrow = [0.0; 4];
        let mut wide = [0.0; 4];
        f32_output.read_into(&mut narrow).unwrap();
        f64_output.read_into(&mut wide).unwrap();
        assert_eq!(
            narrow.map(f32::to_bits),
            expected.map(|value| f32::from(value).to_bits())
        );
        assert_eq!(
            wide.map(f64::to_bits),
            expected.map(|value| f64::from(value).to_bits())
        );
    }
}

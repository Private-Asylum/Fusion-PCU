use super::source;
use fusion_pcu_mlx::MlxRuntime;
use std::sync::Arc;

pub fn run() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let captured = source::capture::<2>();
    let prepared = session
        .prepare_program(Arc::clone(captured.program()))
        .unwrap();
    let [a, b] = prepared.matmul().plan().inputs();
    drop(captured);
    drop(runtime);
    let mut retained = Vec::new();
    for phase in [0.0_f32, 1.0] {
        let left = [1.0 + phase, 2.0, 3.0, 4.0];
        let right = [5.0, 6.0, 7.0, 8.0];
        let lhs = session.upload_f32([2, 2], &left).unwrap();
        let rhs = session.upload_f32([2, 2], &right).unwrap();
        let output = session
            .execute_program(&prepared, &[(a, &lhs), (b, &rhs)])
            .unwrap();
        let mut host = [-73.0_f32; 5];
        output.read_into_f32(&mut host).unwrap();
        for row in 0..2 {
            for column in 0..2 {
                let oracle: f32 = (0..2)
                    .map(|inner| left[row * 2 + inner] * right[inner * 2 + column])
                    .sum();
                assert_eq!(host[row * 2 + column].to_bits(), oracle.to_bits());
            }
        }
        assert_eq!(host[4].to_bits(), (-73.0_f32).to_bits());
        println!(
            "backend={:?} changing_phase={phase} output={:?}",
            session.facts().backend,
            &host[..4]
        );
        retained.push((output, host));
    }
    assert_eq!(prepared.matmul().compilation_trace_count(), 1);
    drop(prepared);
    drop(session);
    for (output, expected) in retained {
        let mut actual = [-73.0_f32; 5];
        output.read_into_f32(&mut actual).unwrap();
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }
    println!(
        "capture/prepared replay, complete oracle, one trace and escaped output owners passed"
    );
}

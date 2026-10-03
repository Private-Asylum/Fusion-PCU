//! Matched copied-host inputs, terminal completion, fresh output and explicit host publication.
use super::activity;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    global::{PcuBackendChoice, PcuExecutionPolicy, PcuSourceShape},
    dialect::tensor::Graph,
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuScalarType,
};
use criterion::Criterion;
#[rustfmt::skip]
use std::{
    hint::black_box,
    sync::Arc,
};
use super::source;

#[allow(clippy::too_many_lines)] // Matched cold plans, full value oracles and identical publication boundaries stay together.
pub fn run<const N: usize>(criterion: &mut Criterion) {
    activity::gpu_idle_guard();
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Mlx,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let session = fusion_pcu_mlx::MlxRuntime::load_default()
        .unwrap()
        .open_gpu(0)
        .unwrap();
    let shapes = [PcuSourceShape::FixedMatrix {
        rows: N,
        columns: N,
    }; 2];
    let captured = global::__pcu_capture_tensor_program(
        shapes,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        source::matrix::__pcu_capture_entry::<N, N, N>,
    )
    .unwrap();
    let prepared = session
        .prepare_program(Arc::clone(captured.program()))
        .unwrap();
    let mut graph = Graph::try_new().unwrap();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        ..PcuNumericalOptions::default()
    });
    let left = graph.input([N, N], PcuScalarType::F32).unwrap();
    let right = graph.input([N, N], PcuScalarType::F32).unwrap();
    let product = graph.matmul(left, right).unwrap();
    let neutral = session
        .prepare_matmul(&graph, graph.node(product).unwrap())
        .unwrap();
    let native = session
        .prepare_native_matmul_control([N, N], [N, N])
        .unwrap();
    let banks = [1.0_f32, 2.0, 3.0].map(|phase| ([[phase; N]; N], [[4.0 - phase; N]; N]));
    let names = [
        "ordinary_pcu",
        "captured_pcu",
        "neutral_graph",
        "native_retained",
        #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "vulkan")))]
        "automatic_pcu",
    ];
    let mut host = vec![77.0_f32; N * N + 1];
    {
        let mut execute = |route: usize, bank: usize, verify: bool| {
            let (left, right) = &banks[bank];
            match route {
                0 | 4 => source::matrix(left, right)
                    .unwrap()
                    .read_into(&mut host)
                    .unwrap(),
                1 => {
                    let [a, b] = prepared.matmul().plan().inputs();
                    prepared
                        .execute_host_into(
                            &[(a, left.as_flattened()), (b, right.as_flattened())],
                            &mut host,
                        )
                        .unwrap();
                }
                2 | 3 => {
                    let a = session.upload_f32([N, N], left.as_flattened()).unwrap();
                    let b = session.upload_f32([N, N], right.as_flattened()).unwrap();
                    let result = if route == 2 {
                        session.execute_matmul(&neutral, &a, &b).unwrap()
                    } else {
                        session
                            .execute_native_matmul_control(&native, &a, &b)
                            .unwrap()
                    };
                    result.read_into_f32(&mut host).unwrap();
                }
                _ => unreachable!("registered matched route"),
            }
            if verify {
                let expected = f32::from(u16::try_from(N).unwrap()) * left[0][0] * right[0][0];
                assert!(
                    host[..N * N]
                        .iter()
                        .all(|value| value.to_bits() == expected.to_bits())
                );
                assert_eq!(host[N * N].to_bits(), 77.0_f32.to_bits());
            }
            black_box(&host);
        };
        for route in 0..names.len() {
            configure_route(route);
            for bank in 0..banks.len() {
                execute(route, bank, true);
            }
        }
        let mut group = criterion.benchmark_group(format!(
            "mlx_ordinary_source/{N}/copied_host_terminal_readback"
        ));
        for (route, name) in names.into_iter().enumerate() {
            configure_route(route);
            // Every separately filtered registration starts with a fully checked physical bank.
            execute(route, 0, true);
            let mut bank = 0;
            group.bench_function(name, |bencher| {
                bencher.iter(|| {
                    bank = (bank + 1) % banks.len();
                    execute(route, bank, false);
                });
            });
        }
        group.finish();
    }
    for (left, right) in &banks {
        let expected = f32::from(u16::try_from(N).unwrap()) * left[0][0] * right[0][0];
        let output = source::matrix(left, right).unwrap();
        output.read_into(&mut host).unwrap();
        assert!(
            host[..N * N]
                .iter()
                .all(|value| value.to_bits() == expected.to_bits())
        );
    }
    assert_eq!(prepared.matmul().compilation_trace_count(), 1);
    assert_eq!(neutral.compilation_trace_count(), 1);
    assert_eq!(native.compilation_trace_count(), 1);
    global::clear_thread_cache().unwrap();
    activity::gpu_post_guard();
}

fn configure_route(route: usize) {
    global::configure(PcuExecutionPolicy {
        backend: if route == 4 {
            PcuBackendChoice::Automatic
        } else {
            PcuBackendChoice::Mlx
        },
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
}

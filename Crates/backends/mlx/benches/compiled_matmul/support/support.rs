//! Captured source/graph replay beside native retained, frontend, and public-C compile controls.
//! This benchmark does not execute ordinary global/direct source calls.

#[path = "activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../census.rs"]
mod census;
#[path = "paired/paired.rs"]
mod paired;
#[rustfmt::skip]
pub use activity::{
    gpu_idle_guard,
    gpu_post_guard,
};
use super::source;

#[rustfmt::skip]
use criterion::{
    Criterion,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxArray,
    MlxNativeMatmulControl,
    MlxPreparedMatmul,
    MlxPreparedProgram,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCompoundArithmeticPolicy,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{
    Graph,
    ValueId,
};

#[cfg(not(feature = "allocation-census"))]
#[rustfmt::skip]
use std::{
    hint::black_box,
    time::Duration,
};

const ROUTE_NAMES: [&str; 5] = [
    "annotated_capture_prepared",
    "explicit_graph_prepared",
    "native_c_retained_primitive",
    "native_c_upstream_frontend",
    "native_c_public_compiled_wrapper",
];
const ROUTES: usize = ROUTE_NAMES.len();

enum Prepared {
    Annotated(MlxPreparedProgram),
    Graph(MlxPreparedMatmul),
    Native(MlxNativeMatmulControl),
    Frontend,
    Compiled(MlxNativeMatmulControl),
}

impl Prepared {
    fn execute(&self, session: &MlxSession, a: &MlxArray, b: &MlxArray) -> MlxArray {
        match self {
            Self::Annotated(prepared) => {
                let [left, right] = prepared.matmul().plan().inputs();
                session
                    .execute_program(prepared, &[(left, a), (right, b)])
                    .unwrap()
            }
            Self::Graph(prepared) => session.execute_matmul(prepared, a, b).unwrap(),
            Self::Native(prepared) => session
                .execute_native_matmul_control(prepared, a, b)
                .unwrap(),
            Self::Frontend => session.execute_native_frontend_control(a, b).unwrap(),
            Self::Compiled(prepared) => session
                .execute_native_compiled_control(prepared, a, b)
                .unwrap(),
        }
    }
    fn traces(&self) -> Option<usize> {
        match self {
            Self::Annotated(prepared) => Some(prepared.matmul().compilation_trace_count()),
            Self::Graph(prepared) => Some(prepared.compilation_trace_count()),
            Self::Native(prepared) | Self::Compiled(prepared) => {
                Some(prepared.compilation_trace_count())
            }
            Self::Frontend => None,
        }
    }
}

fn graph<const N: usize>() -> (Graph, ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        ..PcuNumericalOptions::default()
    });
    let left = graph.input([N, N], PcuScalarType::F32).unwrap();
    let right = graph.input([N, N], PcuScalarType::F32).unwrap();
    let output = graph.matmul(left, right).unwrap();
    (graph, output)
}

fn prepare<const N: usize>(session: &MlxSession, route: usize) -> Prepared {
    match route {
        0 => {
            let captured = source::capture::<N>();
            Prepared::Annotated(
                session
                    .prepare_program(std::sync::Arc::clone(captured.program()))
                    .unwrap(),
            )
        }
        1 => {
            let (graph, output) = graph::<N>();
            Prepared::Graph(
                session
                    .prepare_matmul(&graph, graph.node(output).unwrap())
                    .unwrap(),
            )
        }
        2 => Prepared::Native(
            session
                .prepare_native_matmul_control([N, N], [N, N])
                .unwrap(),
        ),
        3 => Prepared::Frontend,
        4 => Prepared::Compiled(
            session
                .prepare_native_matmul_control([N, N], [N, N])
                .unwrap(),
        ),
        _ => unreachable!(),
    }
}

type HostBank = (Vec<f32>, Vec<f32>, f32);

fn verify<const N: usize>(
    plans: &[Prepared],
    hosts: &[HostBank],
    arrays: &[(MlxArray, MlxArray)],
    session: &MlxSession,
) {
    // Full changing-input oracle around measurement, with earlier outputs retained across replay.
    let mut escaped = Vec::new();
    for (name, plan) in ROUTE_NAMES.iter().zip(plans) {
        if let Some(traces) = plan.traces() {
            assert_eq!(traces, 1);
        }
        for (index, (a, b)) in arrays.iter().enumerate() {
            let result = plan.execute(session, a, b);
            let expected: f32 = (0..N).map(|_| hosts[index].2).sum();
            escaped.push((result, expected));
        }
        if let Some(traces) = plan.traces() {
            assert_eq!(traces, 1);
            eprintln!("MLX_TRACE n={N} route={name} cold_and_current={traces}");
        } else {
            eprintln!("MLX_TRACE n={N} route={name} frontend_no_compiler_trace");
        }
    }
    for (output, expected) in escaped {
        let mut actual = vec![-73.0; N * N + 1];
        output.read_into_f32(&mut actual).unwrap();
        assert!(
            actual[..N * N]
                .iter()
                .all(|value| value.to_bits() == expected.to_bits())
        );
        assert_eq!(actual[N * N].to_bits(), (-73.0_f32).to_bits());
    }
    eprintln!(
        "MLX_BOUNDARY n={N}: logical_result=1 eval=1 stream_sync=1 wait=1; retained routes rebind primitive, frontend constructs graph, compiled route applies public cache wrapper; full_host adds copied_inputs=2 host_materialization=1; driver/C++ allocations remain uncounted"
    );
}

#[cfg_attr(
    feature = "allocation-census",
    allow(
        clippy::needless_pass_by_ref_mut,
        reason = "The shared Criterion harness signature stays identical in the untimed census build."
    )
)]
pub fn benchmark_size<const N: usize>(criterion: &mut Criterion, session: &MlxSession) {
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    let plans: Vec<_> = (0..ROUTES)
        .map(|route| prepare::<N>(session, route))
        .collect();
    let names = ROUTE_NAMES;
    eprintln!(
        "MLX_COLD_SETUP n={N}: annotated includes capture/admission; graph includes graph/admission; retained/compiled include public compile and retained-control setup; frontend is enum-only setup"
    );
    let hosts: Vec<_> = [1.0_f32, 2.0, 3.0]
        .into_iter()
        .map(|phase| {
            (
                vec![phase; N * N],
                vec![4.0 - phase; N * N],
                phase * (4.0 - phase),
            )
        })
        .collect();
    let arrays: Vec<_> = hosts
        .iter()
        .map(|(a, b, _)| {
            (
                session.upload_f32([N, N], a).unwrap(),
                session.upload_f32([N, N], b).unwrap(),
            )
        })
        .collect();
    verify::<N>(&plans, &hosts, &arrays, session);
    #[cfg(feature = "allocation-census")]
    {
        for (route, (name, plan)) in names.iter().zip(&plans).enumerate() {
            let ((), cold) = census::measure(|| drop(prepare::<N>(session, route)));
            let ((), reused) =
                census::measure(|| drop(plan.execute(session, &arrays[1].0, &arrays[1].1)));
            let mut host = vec![0.0; N * N];
            let ((), full) = census::measure(|| {
                let a = session.upload_f32([N, N], &hosts[1].0).unwrap();
                let b = session.upload_f32([N, N], &hosts[1].1).unwrap();
                let output = plan.execute(session, &a, &b);
                output.read_into_f32(&mut host).unwrap();
            });
            eprintln!(
                "MLX_RUST_CENSUS n={N} route={name} counts=[allocations,bytes,frees] cold={cold:?} reused={reused:?} full_host={full:?}"
            );
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    {
        let mut group = criterion.benchmark_group(format!("mlx_compiled_matmul/{N}"));
        group
            .sample_size(30)
            .warm_up_time(Duration::from_millis(250))
            .measurement_time(Duration::from_secs(1));
        for (route, (name, plan)) in names.iter().zip(&plans).enumerate() {
            let mut index = 0;
            group.bench_function(format!("preuploaded_reused_inputs/{name}"), |bencher| {
                bencher.iter(|| {
                    index = (index + 1) % arrays.len();
                    drop(black_box(plan.execute(
                        session,
                        &arrays[index].0,
                        &arrays[index].1,
                    )));
                });
            });
            let mut host = vec![0.0; N * N];
            group.bench_function(
                format!("copied_host_inputs_terminal_readback/{name}"),
                |bencher| {
                    bencher.iter(|| {
                        index = (index + 1) % hosts.len();
                        let a = session.upload_f32([N, N], &hosts[index].0).unwrap();
                        let b = session.upload_f32([N, N], &hosts[index].1).unwrap();
                        let output = plan.execute(session, &a, &b);
                        output.read_into_f32(&mut host).unwrap();
                        black_box(&host);
                    });
                },
            );
            group.bench_function(format!("cold_capture_or_native_setup/{name}"), |bencher| {
                bencher.iter(|| drop(black_box(prepare::<N>(session, route))));
            });
        }
        group.finish();
    }
    verify::<N>(&plans, &hosts, &arrays, session);
}

pub use paired::run as paired_run;

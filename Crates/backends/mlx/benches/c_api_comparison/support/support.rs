//! Diagnostic capture/admission and native pairs; ordinary facade execution is not exercised.

use super::source;
#[path = "../../compiled_matmul/support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../compiled_matmul/census.rs"]
mod census;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    c_api_evaluation as c,
    MlxArray,
    MlxMatmulPlan,
    MlxNativeMatmulControl,
    MlxPreparedMatmul,
    MlxPreparedProgram,
    MlxRuntime,
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
    TensorOwnedSelectedProgram,
    ValueId,
};
use criterion::Criterion;
use std::sync::Arc;
#[cfg(not(feature = "allocation-census"))]
#[rustfmt::skip]
use std::{
    hint::black_box,
    time::{
        Duration,
        Instant,
    },
};

const NAMES: [&str; 8] = [
    "cpp_annotated_capture_retained",
    "cpp_explicit_graph_retained",
    "cpp_native_retained",
    "cpp_native_frontend",
    "c_annotated_capture_admitted_frontend",
    "c_explicit_graph_admitted_frontend",
    "c_native_frontend",
    "c_native_compiled_wrapper",
];

struct Context {
    cpp: MlxSession,
    c: c::Session,
}
struct Admission {
    plan: MlxMatmulPlan,
    _program: Option<Arc<TensorOwnedSelectedProgram>>,
}
enum Prepared {
    Source(MlxPreparedProgram),
    Graph(MlxPreparedMatmul),
    Native(MlxNativeMatmulControl),
    CppFrontend,
    CAdmitted(Admission),
    CFrontend,
    CCompiled(c::Compiled),
}
enum Array {
    Cpp(MlxArray),
    C(c::Array),
}
impl Array {
    fn read(&self, host: &mut [f32]) {
        match self {
            Self::Cpp(array) => array.read_into_f32(host).unwrap(),
            Self::C(array) => array.read(host).unwrap(),
        }
    }
}
impl Prepared {
    fn upload<const N: usize>(&self, context: &Context, host: &[f32]) -> Array {
        if matches!(
            self,
            Self::CAdmitted(_) | Self::CFrontend | Self::CCompiled(_)
        ) {
            Array::C(context.c.upload([N, N], host).unwrap())
        } else {
            Array::Cpp(context.cpp.upload_f32([N, N], host).unwrap())
        }
    }
    fn execute(&self, context: &Context, left: &Array, right: &Array) -> Array {
        match (self, left, right) {
            (Self::Source(prepared), Array::Cpp(left), Array::Cpp(right)) => {
                let [a, b] = prepared.matmul().plan().inputs();
                Array::Cpp(
                    context
                        .cpp
                        .execute_program(prepared, &[(a, left), (b, right)])
                        .unwrap(),
                )
            }
            (Self::Graph(prepared), Array::Cpp(left), Array::Cpp(right)) => {
                Array::Cpp(context.cpp.execute_matmul(prepared, left, right).unwrap())
            }
            (Self::Native(prepared), Array::Cpp(left), Array::Cpp(right)) => Array::Cpp(
                context
                    .cpp
                    .execute_native_matmul_control(prepared, left, right)
                    .unwrap(),
            ),
            (Self::CppFrontend, Array::Cpp(left), Array::Cpp(right)) => Array::Cpp(
                context
                    .cpp
                    .execute_native_direct_matmul_control(left, right)
                    .unwrap(),
            ),
            (Self::CAdmitted(admission), Array::C(left), Array::C(right)) => {
                // Actual source/graph provenance is frozen cold. SDK frontend metadata is new.
                assert_ne!(admission.plan.inputs()[0], admission.plan.inputs()[1]);
                Array::C(context.c.direct(left, right).unwrap())
            }
            (Self::CFrontend, Array::C(left), Array::C(right)) => {
                Array::C(context.c.direct(left, right).unwrap())
            }
            (Self::CCompiled(prepared), Array::C(left), Array::C(right)) => {
                Array::C(prepared.execute(left, right).unwrap())
            }
            _ => unreachable!("Each route retains exact native session affinity"),
        }
    }
    fn traces(&self) -> Option<usize> {
        match self {
            Self::Source(prepared) => Some(prepared.matmul().compilation_trace_count()),
            Self::Graph(prepared) => Some(prepared.compilation_trace_count()),
            Self::Native(prepared) => Some(prepared.compilation_trace_count()),
            Self::CCompiled(prepared) => Some(prepared.traces()),
            _ => None,
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
fn prepare<const N: usize>(context: &Context, route: usize) -> Prepared {
    match route {
        0 => Prepared::Source(
            context
                .cpp
                .prepare_program(Arc::clone(source::capture::<N>().program()))
                .unwrap(),
        ),
        1 => {
            let (graph, output) = graph::<N>();
            Prepared::Graph(
                context
                    .cpp
                    .prepare_matmul(&graph, graph.node(output).unwrap())
                    .unwrap(),
            )
        }
        2 => Prepared::Native(
            context
                .cpp
                .prepare_native_matmul_control([N, N], [N, N])
                .unwrap(),
        ),
        3 => Prepared::CppFrontend,
        4 => {
            let program = Arc::clone(source::capture::<N>().program());
            let plan = MlxMatmulPlan::assess_program(&program).unwrap();
            assert_eq!(plan.shape(), [N, N]);
            Prepared::CAdmitted(Admission {
                plan,
                _program: Some(program),
            })
        }
        5 => {
            let (graph, output) = graph::<N>();
            let plan = MlxMatmulPlan::assess(&graph, graph.node(output).unwrap()).unwrap();
            Prepared::CAdmitted(Admission {
                plan,
                _program: None,
            })
        }
        6 => Prepared::CFrontend,
        7 => Prepared::CCompiled(context.c.prepare([N, N], [N, N]).unwrap()),
        _ => unreachable!(),
    }
}
type Bank = (Vec<f32>, Vec<f32>, f32);
fn verify<const N: usize>(
    context: &Context,
    plans: &[Prepared],
    hosts: &[Bank],
    arrays: &[Vec<(Array, Array)>],
) {
    let mut escaped = Vec::new();
    for (plan, arrays) in plans.iter().zip(arrays) {
        if let Some(traces) = plan.traces() {
            assert_eq!(traces, 1);
        }
        for (index, (left, right)) in arrays.iter().enumerate() {
            let output = plan.execute(context, left, right);
            let expected: f32 = (0..N).map(|_| hosts[index].2).sum();
            escaped.push((output, expected));
        }
    }
    for (output, expected) in escaped {
        let mut host = vec![-73.0; N * N + 1];
        output.read(&mut host);
        assert!(
            host[..N * N]
                .iter()
                .all(|value| value.to_bits() == expected.to_bits())
        );
        assert_eq!(host[N * N].to_bits(), (-73.0_f32).to_bits());
    }
    for (route, plan) in plans.iter().enumerate() {
        eprintln!(
            "MLX_C_TRACE n={N} route={} traces={:?}",
            NAMES[route],
            plan.traces()
        );
    }
}
fn full<const N: usize>(context: &Context, plan: &Prepared, bank: &Bank, host: &mut [f32]) {
    let left = plan.upload::<N>(context, &bank.0);
    let right = plan.upload::<N>(context, &bank.1);
    let output = plan.execute(context, &left, &right);
    output.read(host);
}
#[cfg(not(feature = "allocation-census"))]
fn paired<const N: usize, const HOST: bool>(
    context: &Context,
    plans: &[Prepared],
    hosts: &[Bank],
    arrays: &[Vec<(Array, Array)>],
) {
    const ROUNDS: usize = 48;
    const CALLS: usize = 16;
    let mut host = vec![0.0; N * N];
    let boundary = if HOST {
        "full_host"
    } else {
        "preuploaded_terminal_drop"
    };
    let mut samples = Vec::with_capacity(ROUNDS * plans.len());
    for round in 0..ROUNDS {
        let mut order: Vec<_> = (0..plans.len())
            .map(|offset| (round + offset) % plans.len())
            .collect();
        if round / plans.len() % 2 == 1 {
            order.reverse();
        }
        for (position, &route) in order.iter().enumerate() {
            let start = Instant::now();
            for call in 0..CALLS {
                let index = (round + call) % hosts.len();
                if HOST {
                    full::<N>(context, &plans[route], &hosts[index], &mut host);
                    black_box(&host);
                } else {
                    drop(black_box(plans[route].execute(
                        context,
                        &arrays[route][index].0,
                        &arrays[route][index].1,
                    )));
                }
            }
            samples.push((round, position, route, start.elapsed().as_nanos()));
        }
    }
    for (round, position, route, nanoseconds) in samples {
        eprintln!(
            "MLX_C_PAIRED {{\"n\":{N},\"boundary\":\"{boundary}\",\"round\":{round},\"position\":{position},\"route\":\"{}\",\"calls\":{CALLS},\"nanoseconds\":{nanoseconds}}}",
            NAMES[route]
        );
    }
}
#[cfg_attr(
    feature = "allocation-census",
    allow(
        clippy::needless_pass_by_ref_mut,
        reason = "Criterion harness signature stays identical in the untimed census."
    )
)]
fn size<const N: usize>(criterion: &mut Criterion, context: &Context) {
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    let plans: Vec<_> = (0..NAMES.len())
        .map(|route| prepare::<N>(context, route))
        .collect();
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
    let arrays: Vec<Vec<_>> = plans
        .iter()
        .map(|plan| {
            hosts
                .iter()
                .map(|(a, b, _)| (plan.upload::<N>(context, a), plan.upload::<N>(context, b)))
                .collect()
        })
        .collect();
    verify::<N>(context, &plans, &hosts, &arrays);
    #[cfg(feature = "allocation-census")]
    for (route, plan) in plans.iter().enumerate() {
        let ((), cold) = census::measure(|| drop(prepare::<N>(context, route)));
        let ((), resident) = census::measure(|| {
            drop(plan.execute(context, &arrays[route][1].0, &arrays[route][1].1));
        });
        let mut host = vec![0.0; N * N];
        let ((), full_host) = census::measure(|| full::<N>(context, plan, &hosts[1], &mut host));
        eprintln!(
            "MLX_C_RUST_CENSUS n={N} route={} cold={cold:?} resident={resident:?} full_host={full_host:?}; C++/SDK allocations uncounted",
            NAMES[route]
        );
    }
    #[cfg(not(feature = "allocation-census"))]
    {
        if std::env::var_os("PCU_MLX_C_PAIRED_DIAGNOSTIC").is_some() {
            paired::<N, false>(context, &plans, &hosts, &arrays);
            paired::<N, true>(context, &plans, &hosts, &arrays);
        } else {
            let mut group = criterion.benchmark_group(format!("mlx_c_api_comparison/{N}"));
            group
                .sample_size(30)
                .warm_up_time(Duration::from_millis(250))
                .measurement_time(Duration::from_secs(1));
            for (route, plan) in plans.iter().enumerate() {
                let mut index = 0;
                group.bench_function(
                    format!("preuploaded_terminal_drop/{}", NAMES[route]),
                    |bencher| {
                        bencher.iter(|| {
                            index = (index + 1) % hosts.len();
                            drop(black_box(plan.execute(
                                context,
                                &arrays[route][index].0,
                                &arrays[route][index].1,
                            )));
                        });
                    },
                );
                let mut host = vec![0.0; N * N];
                group.bench_function(
                    format!("copied_host_terminal_readback_drop/{}", NAMES[route]),
                    |bencher| {
                        bencher.iter(|| {
                            index = (index + 1) % hosts.len();
                            full::<N>(context, plan, &hosts[index], &mut host);
                            black_box(&host);
                        });
                    },
                );
            }
            // Frontend controls have no compiled artifact; their enum-only setup is named
            // honestly. Source cold capture/admission and C compile allocations stay visible.
            for (route, name) in NAMES.iter().enumerate() {
                group.bench_function(format!("cold_capture_or_control_setup/{name}"), |bencher| {
                    bencher.iter(|| drop(black_box(prepare::<N>(context, route))));
                });
            }
            group.finish();
        }
    }
    verify::<N>(context, &plans, &hosts, &arrays);
}
pub fn comparison(criterion: &mut Criterion) {
    activity::gpu_idle_guard();
    let cpp =
        MlxRuntime::load(std::env::var_os("PCU_MLX_BRIDGE").expect("set exact owned bridge path"))
            .unwrap();
    let c = c::Runtime::load(
        std::env::var_os("PCU_MLX_C_EVALUATION").expect("set safety-patched isolated C library"),
    )
    .unwrap();
    let context = Context {
        cpp: cpp.open_gpu(0).unwrap(),
        c: c.open_gpu(0).unwrap(),
    };
    eprintln!(
        "MLX_C_COMPARISON sdk={} private_C_ABI=3 safety_patched_C_subset=true; all_routes eval=1 explicit_stream_sync=1 wait=1 output_drop=1; full_host copied_inputs=2 host_readback=1; C_compiled_wrapper_cache_remains",
        cpp.version()
    );
    size::<32>(criterion, &context);
    size::<128>(criterion, &context);
    drop(context);
    activity::gpu_post_guard();
}

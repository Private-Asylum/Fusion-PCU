//! Actual source, detached scalar/SIMD plans and independent base-256 native transactions.
#[path = "../low_precision/ffi/ffi.rs"]
mod heap;
#[path = "../../tests/wide_integer/oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: heap::CountingAllocator = heap::CountingAllocator;
#[rustfmt::skip]
use criterion::{
    Criterion,
    BenchmarkId,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuDispatchIntegerBinaryOp,
    PcuRangePolicy,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuHostBackend,
    PcuCpuPreparedHost,
};
use oracle::Wide;
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
const N: usize = 65;
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn evaluate<T: Wide>(
    kind: usize,
    lane: usize,
    a: &[T],
    b: &[T],
) -> Result<T, PcuExecutionFaultKind> {
    let (op, left, right) = match kind {
        0 => (PcuDispatchIntegerBinaryOp::Add, a[lane], a[lane]),
        1 => (PcuDispatchIntegerBinaryOp::Mul, a[lane], a[lane]),
        2 => (PcuDispatchIntegerBinaryOp::Sub, a[lane], b[lane]),
        3 => (PcuDispatchIntegerBinaryOp::Sub, a[lane], a[0]),
        4 => (PcuDispatchIntegerBinaryOp::Sub, a[0], a[lane]),
        _ => unreachable!(),
    };
    oracle::evaluate(left, right, op)
}
fn native<T: Wide>(
    kind: usize,
    range: PcuRangePolicy,
    a: &[T],
    b: &[T],
    out: &mut [T],
) -> Result<(), PcuExecutionFault> {
    assert!(a.len() >= N && b.len() >= N && out.len() >= N);
    if range == PcuRangePolicy::Reject {
        for lane in 0..N {
            evaluate(kind, lane, a, b).map_err(|kind| PcuExecutionFault {
                kind,
                recovered: false,
                invocation_id: u64::try_from(lane).unwrap(),
            })?;
        }
    }
    let mut first = None;
    for (lane, value) in out.iter_mut().take(N).enumerate() {
        *value = match evaluate(kind, lane, a, b) {
            Ok(v) => v,
            Err(kind) => {
                first.get_or_insert_with(|| PcuExecutionFault {
                    kind,
                    recovered: true,
                    invocation_id: u64::try_from(lane).unwrap(),
                });
                if kind == PcuExecutionFaultKind::ArithmeticUnderflow {
                    oracle::minimum::<T>()
                } else {
                    oracle::maximum::<T>()
                }
            }
        };
    }
    first.map_or(Ok(()), Err)
}
fn explicit<T: Wide>(
    kind: usize,
    plan: &mut PcuCpuPreparedHost,
    a: &[T],
    b: &[T],
    out: &mut [T],
) -> Result<(), PcuExecutionFault> {
    let target = |slot| PcuBindingRef::new(0, slot);
    let result = match kind {
        0 => plan.call(&mut [
            PcuHostArgument::read_write(target(0), out),
            PcuHostArgument::read(target(1), a),
        ]),
        1 | 4 => plan.call(&mut [
            PcuHostArgument::read(target(0), &[] as &[T]),
            PcuHostArgument::read_write(target(1), out),
            PcuHostArgument::read(target(2), a),
        ]),
        2 => plan.call(&mut [
            PcuHostArgument::read(target(0), b),
            PcuHostArgument::read_write(target(1), out),
            PcuHostArgument::read(target(2), a),
        ]),
        3 => plan.call(&mut [
            PcuHostArgument::read(target(0), a),
            PcuHostArgument::read_write(target(1), out),
        ]),
        _ => unreachable!(),
    };
    result.map_err(|error| error.fault().unwrap())
}
fn compare<T: Wide>(
    c: &mut Criterion,
    kind: usize,
    range: PcuRangePolicy,
    mut selected: PcuCpuPreparedHost,
    mut scalar: PcuCpuPreparedHost,
    mut ordinary: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    global::clear_thread_cache().unwrap();
    let sentinel = oracle::small::<T>(77);
    let mut banks: [[T; N]; 2] = std::array::from_fn(|phase| {
        std::array::from_fn(|lane| {
            oracle::small::<T>(
                u8::try_from(if kind == 4 {
                    10 + phase - lane % 7
                } else {
                    2 + phase + lane % 7
                })
                .unwrap(),
            )
        })
    });
    match kind {
        0 | 1 | 4 => banks[1][2] = oracle::maximum::<T>(),
        _ => banks[1][2] = oracle::minimum::<T>(),
    }
    let b = [oracle::small::<T>(1); N];
    let mut expected = [[sentinel; N + 3]; 2];
    let notices = std::array::from_fn::<_, 2, _>(|bank| {
        native(kind, range, &banks[bank], &b, &mut expected[bank])
    });
    let mut output = [sentinel; N + 3];
    let mut run = |route, bank: usize| {
        output.fill(sentinel);
        let notice = match route {
            0 => ordinary(&banks[bank], &b, &mut output),
            1 => explicit(kind, &mut selected, &banks[bank], &b, &mut output),
            2 => explicit(kind, &mut scalar, &banks[bank], &b, &mut output),
            _ => native(kind, range, &banks[bank], &b, &mut output),
        };
        assert_eq!(notice, notices[bank]);
        assert_eq!(output, expected[bank]);
    };
    let mut group = c.benchmark_group(format!(
        "cpu_integer_operands/{:?}/{kind}/{range:?}",
        T::TYPE
    ));
    if portable() {
        println!("PortableV1 cold admitted {:?}/{kind}/{range:?}", T::TYPE);
    }
    for (route, label) in [
        "source_ordinary",
        "prepared_selected",
        "graph_scalar",
        "independent_native_checked",
    ]
    .into_iter()
    .enumerate()
    {
        run(route, 0);
        run(route, 1);
        let scores = SCORES.load(Ordering::Relaxed);
        let counts = heap::count_heap(|| {
            for bank in 0..64 {
                run(route, bank & 1);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(SCORES.load(Ordering::Relaxed), scores);
        println!(
            "caller census {:?}/{kind}/{range:?}/{label}:64 changing valid/fault banks,Rustalloc {} reallocation {} free {}",
            T::TYPE,
            counts.allocations,
            counts.reallocations,
            counts.frees
        );
        group.bench_function(BenchmarkId::new(label, N), |bench| {
            bench.iter(|| run(route, std::hint::black_box(1)));
        });
    }
    group.finish();
}
trait BuildPlan {
    fn prepare(&self, backend: &PcuCpuHostBackend) -> PcuCpuPreparedHost;
}
impl<const M: usize> BuildPlan for fusion_pcu_core::model::PcuDispatchKernelBuilder<'_, M> {
    fn prepare(&self, backend: &PcuCpuHostBackend) -> PcuCpuPreparedHost {
        {
            let mut ir = self.ir();
            numerical(&mut ir);
            backend.prepare_host_kernel(&ir).unwrap()
        }
    }
}
impl<const M: usize> BuildPlan for fusion_pcu_core::model::PcuGridStrideKernelBuilder<'_, M> {
    fn prepare(&self, backend: &PcuCpuHostBackend) -> PcuCpuPreparedHost {
        self.with_ir(|ir| {
            let mut ir = *ir;
            numerical(&mut ir);
            backend.prepare_host_kernel(&ir).unwrap()
        })
    }
}
// This optional proof mode is read only during cold preparation; warm workloads are identical.
fn portable() -> bool {
    std::env::var_os("PCU_INTEGER_PORTABLE").is_some()
}
fn numerical(ir: &mut pcu_facade::PcuDispatchKernelIr<'_>) {
    if portable() {
        ir.numerical_requirements.numerical_options.reproducibility =
            pcu_facade::PcuReproducibility::PortableV1;
    }
}
fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        numerical_options: pcu_facade::PcuNumericalOptions {
            reproducibility: if portable() {
                pcu_facade::PcuReproducibility::PortableV1
            } else {
                pcu_facade::PcuReproducibility::Unspecified
            },
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
}
fn width<T: Wide>(c: &mut Criterion) {
    configure();
    let selected = PcuCpuHostBackend::detect();
    let scalar = PcuCpuHostBackend::scalar();
    macro_rules! profile {
        ($name:ident,$bindings:ident,$ir:ident,$kind:expr,$range:ident,$call:expr) => {{
            let bindings = source::$bindings::<T>();
            let graph = source::$ir::<T, N>(&bindings).unwrap();
            compare::<T>(
                c,
                $kind,
                PcuRangePolicy::$range,
                graph.prepare(&selected),
                graph.prepare(&scalar),
                $call,
            );
        }};
    }
    profile!(
        doubled,
        doubled_bindings,
        doubled_ir,
        0,
        Reject,
        |a, _b, o| source::doubled::<T, N>(o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
    profile!(
        squared,
        squared_bindings,
        squared_ir,
        1,
        Reject,
        |a, _b, o| source::squared::<T, N>(&[], o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
    profile!(
        reordered,
        reordered_bindings,
        reordered_ir,
        2,
        Reject,
        |a, b, o| source::reordered::<T, N>(b, o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
    profile!(
        independent,
        independent_bindings,
        independent_ir,
        3,
        Reject,
        |a, _b, o| source::independent::<T, N>(a, o).map_err(|e| e.arithmetic_fault().unwrap())
    );
    profile!(grid, grid_bindings, grid_ir, 4, Reject, |a, _b, o| {
        source::grid::<T, N>(&[], o, a).map_err(|e| e.arithmetic_fault().unwrap())
    });
    profile!(
        doubled_clamp,
        doubled_clamp_bindings,
        doubled_clamp_ir,
        0,
        Clamp,
        |a, _b, o| source::doubled_clamp::<T, N>(o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
    profile!(
        squared_clamp,
        squared_clamp_bindings,
        squared_clamp_ir,
        1,
        Clamp,
        |a, _b, o| source::squared_clamp::<T, N>(&[], o, a)
            .map_err(|e| e.arithmetic_fault().unwrap())
    );
    profile!(
        reordered_clamp,
        reordered_clamp_bindings,
        reordered_clamp_ir,
        2,
        Clamp,
        |a, b, o| source::reordered_clamp::<T, N>(b, o, a)
            .map_err(|e| e.arithmetic_fault().unwrap())
    );
    profile!(
        independent_clamp,
        independent_clamp_bindings,
        independent_clamp_ir,
        3,
        Clamp,
        |a, _b, o| source::independent_clamp::<T, N>(a, o)
            .map_err(|e| e.arithmetic_fault().unwrap())
    );
    profile!(
        grid_clamp,
        grid_clamp_bindings,
        grid_clamp_ir,
        4,
        Clamp,
        |a, _b, o| source::grid_clamp::<T, N>(&[], o, a).map_err(|e| e.arithmetic_fault().unwrap())
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
fn benchmarks(c: &mut Criterion) {
    macro_rules! widths{($($t:ty),+)=>{$(width::<$t>(c);)+};}
    widths!(
        i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, PcuI256, PcuU256, PcuI512, PcuU512
    );
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);

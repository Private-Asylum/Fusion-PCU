//! Fresh host prefixes, retained primitive, private siblings/status, both reads and joint publish.
use std::hint::black_box;
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuScalar,
    PcuCheckedIntegerDivision,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuHostArgument,
    PcuBindingRef,
    PcuImplementationRequirements,
};
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../tensor_binary/census/census.rs"]
mod census;
#[path = "../../../tests/checked_div_rem/graph/graph.rs"]
mod graph;
#[path = "../../../tests/checked_div_rem/source/source.rs"]
mod source;
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    1
}
type Bank<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>);
trait Sample: PcuCheckedIntegerDivision {
    fn raw(value: u64) -> Self;
}
macro_rules! samples {($($ty:ty=>$width:literal;)+) => {$(impl Sample for $ty {
    fn raw(value:u64)->Self {let mut bytes=[0;$width];let n=bytes.len().min(8);bytes[..n].copy_from_slice(&value.to_le_bytes()[..n]);Self::decode_le(bytes)}
})+};}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;pcu_facade::PcuU256=>32;pcu_facade::PcuI256=>32;pcu_facade::PcuU512=>64;pcu_facade::PcuI512=>64;}
fn same<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), actual).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), expected).bytes()
    );
}
#[allow(clippy::too_many_lines)]
// Four retained peers share both explicit prefix/read/publication boundaries and changing-bank oracle.
#[allow(clippy::significant_drop_tightening)]
// Criterion's group is used by every peer, then explicitly finished before clearing the ordinary cache.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Untimed census context preserves the same driver signature.
fn compare<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MlxSession) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    let backend = session.checked_div_rem_backend();
    let mut source = source::direct_prepare::<T, N, _>(&backend).unwrap();
    let mut graph = graph::fixture::<T, _>(
        u32::try_from(N).unwrap(),
        false,
        false,
        PcuImplementationRequirements::default(),
        |ir| backend.prepare_host_kernel(ir),
    )
    .unwrap();
    let mut native = session
        .prepare_checked_div_rem_control(T::TYPE, N, [N; 2], [false; 2])
        .unwrap();
    let banks: [Bank<T>; 64] = std::array::from_fn(|bank| {
        let left: Vec<T> = (0..N)
            .map(|lane| T::raw(u64::try_from((lane ^ bank) % 97 + 1).unwrap()))
            .collect();
        let right: Vec<T> = (0..N)
            .map(|lane| T::raw(u64::try_from((lane + bank) % 5 + 1).unwrap()))
            .collect();
        let q = left
            .iter()
            .zip(&right)
            .map(|(&a, &b)| a.pcu_checked_div(b).unwrap())
            .collect();
        let r = left
            .iter()
            .zip(&right)
            .map(|(&a, &b)| a.pcu_checked_rem(b).unwrap())
            .collect();
        (left, right, q, r)
    });
    let sentinel = T::raw(19);
    let mut q = vec![sentinel; N + 2];
    let mut r = vec![sentinel; N + 2];
    let mut execute = |route: usize, bank: usize, q: &mut [T], r: &mut [T]| {
        let (left, right, _, _) = &banks[bank];
        match route {
            0 => source(
                black_box(left),
                black_box(right),
                black_box(q),
                black_box(r),
            )
            .unwrap(),
            1 => {
                let input = *graph.input_bindings();
                let output = *graph.output_bindings();
                graph
                    .call(&mut [
                        PcuHostArgument::read(input[0], black_box(left)),
                        PcuHostArgument::read(input[1], black_box(right)),
                        PcuHostArgument::read_write(output[0], black_box(q)),
                        PcuHostArgument::read_write(output[1], black_box(r)),
                    ])
                    .unwrap();
            }
            2 => native
                .call(
                    [black_box(left), black_box(right)],
                    [black_box(q), black_box(r)],
                )
                .unwrap(),
            _ => source::direct::<T, N>(
                black_box(left),
                black_box(right),
                black_box(q),
                black_box(r),
            )
            .unwrap(),
        }
    };
    let verify = |bank: usize, q: &[T], r: &[T]| {
        same(&q[..N], &banks[bank].2);
        same(&r[..N], &banks[bank].3);
        same(&q[N..], &[sentinel; 2]);
        same(&r[N..], &[sentinel; 2]);
    };
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "mlx_checked_div_rem_{:?}_host_joint_publish",
        T::TYPE
    ));
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    for (route, name) in [
        "prepared_source",
        "explicit_ir",
        "direct_native",
        "ordinary_pcu",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in 0..64 {
            execute(route, bank, &mut q, &mut r);
            verify(bank, &q, &r);
        }
        #[cfg(not(feature = "allocation-census"))]
        group.bench_function(BenchmarkId::new(name, N), |bench| {
            let mut bank = 0;
            bench.iter(|| {
                execute(route, bank, &mut q, &mut r);
                black_box((&q, &r));
                bank = (bank + 1) % 64;
            });
        });
        #[cfg(feature = "allocation-census")]
        {
            let before = SCORES.load(std::sync::atomic::Ordering::Relaxed);
            let mut total = census::Census::default();
            for bank in 0..64 {
                let ((), count) = census::measure(|| execute(route, bank, &mut q, &mut r));
                total.alloc_calls += count.alloc_calls;
                total.realloc_calls += count.realloc_calls;
                total.dealloc_calls += count.dealloc_calls;
                total.requested_bytes += count.requested_bytes;
                verify(bank, &q, &r);
            }
            let scored = SCORES.load(std::sync::atomic::Ordering::Relaxed) - before;
            assert_eq!(scored, 0, "warm division call rescored candidates");
            eprintln!(
                "Rust allocation census/div_rem/{:?}/{N}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, invocation_score_callbacks={scored} (both private reads and joint host publish; native/device allocations unknown)",
                T::TYPE,
                total.alloc_calls,
                total.realloc_calls,
                total.dealloc_calls,
                total.requested_bytes
            );
        }
        for bank in 0..64 {
            execute(route, bank, &mut q, &mut r);
            verify(bank, &q, &r);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy::default()).unwrap();
}
pub fn run(criterion: &mut Criterion) {
    if !std::env::args().any(|arg| arg == "--test") {
        activity::guard();
    }
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    macro_rules! run {($($ty:ty),+) => {$(compare::<$ty,65>(criterion,&session);compare::<$ty,4096>(criterion,&session);)+};}
    run!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        pcu_facade::PcuU256,
        pcu_facade::PcuI256,
        pcu_facade::PcuU512,
        pcu_facade::PcuI512
    );
}

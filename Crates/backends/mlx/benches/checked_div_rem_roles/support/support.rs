//! Actual unique prefixes staged once, retained primitive, private reads/cleanup, dual host publish.
use std::hint::black_box;
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxSession,
    MlxHostKernelError,
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
};
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../tensor_binary/census/census.rs"]
mod census;
#[path = "graph/graph.rs"]
mod graph;
#[cfg(not(feature = "portable-joint-control"))]
#[path = "../../../tests/checked_div_rem/roles/source/source.rs"]
mod source;
#[cfg(feature = "portable-joint-control")]
#[path = "../../../tests/checked_div_rem/roles/portable/source/source.rs"]
mod source;
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    1
}
trait Sample: PcuCheckedIntegerDivision {
    fn raw(value: u64) -> Self;
}
macro_rules! samples {($($ty:ty=>$width:literal;)+)=>{$(impl Sample for $ty {
fn raw(value:u64)->Self {let mut b=[0;$width];let n=b.len().min(8);b[..n].copy_from_slice(&value.to_le_bytes()[..n]);Self::decode_le(b)}
})+};}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;pcu_facade::PcuU256=>32;pcu_facade::PcuI256=>32;pcu_facade::PcuU512=>64;pcu_facade::PcuI512=>64;}
type Bank<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>);
fn same<T: PcuScalar>(a: &[T], b: &[T]) {
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), a).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), b).bytes()
    );
}
fn ordinary<T: Sample, const N: usize>(profile: usize, a: &[T], b: &[T], q: &mut [T], r: &mut [T]) {
    match profile {
        0 => source::repeated::<T, N>(q, a, r),
        1 => source::unread::<T, N>(&[], r, a, q),
        2 => source::reordered::<T, N>(b, r, a, q),
        3 => source::grid::<T, N>(&[], r, a, q),
        4 => source::scalar_divisor::<T, N>(q, &b[0], r, a),
        5 => source::scalar_grid::<T, N>(&a[0], q, r),
        _ => unreachable!(),
    }
    .unwrap();
}
#[allow(clippy::too_many_lines)]
// Four independent peers include cold role preparation and the full matched transactional boundary.
#[allow(clippy::significant_drop_tightening)]
// Criterion group remains live through all four peer registrations, then explicit cache cleanup.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Same driver ABI in untimed census mode.
fn compare<T: Sample, const N: usize, F>(
    criterion: &mut Criterion,
    session: &MlxSession,
    profile: usize,
    mut prepared: F,
) where
    F: FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), MlxHostKernelError>,
{
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        #[cfg(feature = "portable-joint-control")]
        numerical_options: pcu_facade::PcuNumericalOptions {
            reproducibility: pcu_facade::PcuReproducibility::PortableV1,
            ..Default::default()
        },
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    let backend = session.checked_div_rem_role_backend();
    let mut ir = graph::fixture::<T, _>(u32::try_from(N).unwrap(), profile, |ir| {
        backend.prepare_host_kernel(ir)
    })
    .unwrap();
    let counts = match profile {
        0 | 1 | 3 => [N, 0],
        2 => [N, N],
        4 => [N, 1],
        5 => [1, 0],
        _ => unreachable!(),
    };
    let input_count = ir.input_bindings().len();
    assert_eq!(input_count, if counts[1] == 0 { 1 } else { 2 });
    let slots = if input_count == 1 { [0, 0] } else { [0, 1] };
    let broadcast = [matches!(profile, 3 | 5), matches!(profile, 1 | 4 | 5)];
    let mut native = session
        .prepare_checked_div_rem_control(T::TYPE, N, slots.map(|slot| counts[slot]), broadcast)
        .unwrap();
    let banks: [Bank<T>; 64] = std::array::from_fn(|bank| {
        let a: Vec<T> = (0..N)
            .map(|lane| T::raw(u64::try_from((lane ^ bank) % 89 + 1).unwrap()))
            .collect();
        let b: Vec<T> = (0..N)
            .map(|lane| T::raw(u64::try_from((lane + bank) % 7 + 1).unwrap()))
            .collect();
        let value = |slot: usize, lane: usize| if slot == 0 { a[lane] } else { b[lane] };
        let operands = |lane: usize| {
            [
                value(slots[0], if broadcast[0] { 0 } else { lane }),
                value(slots[1], if broadcast[1] { 0 } else { lane }),
            ]
        };
        let q = (0..N)
            .map(|lane| {
                let [a, b] = operands(lane);
                a.pcu_checked_div(b).unwrap()
            })
            .collect();
        let r = (0..N)
            .map(|lane| {
                let [a, b] = operands(lane);
                a.pcu_checked_rem(b).unwrap()
            })
            .collect();
        (a, b, q, r)
    });
    let sentinel = T::raw(91);
    let mut q = vec![sentinel; N + 2];
    let mut r = vec![sentinel; N + 2];
    let mut execute = |route: usize, bank: usize, q: &mut [T], r: &mut [T]| {
        let (a, b, _, _) = &banks[bank];
        match route {
            0 => prepared(black_box(a), black_box(b), black_box(q), black_box(r)).unwrap(),
            1 => {
                let inputs = ir.input_bindings();
                let outputs = ir.output_bindings();
                if input_count == 1 {
                    ir.call(&mut [
                        PcuHostArgument::read(inputs[0], black_box(a)),
                        PcuHostArgument::read_write(outputs[0], black_box(q)),
                        PcuHostArgument::read_write(outputs[1], black_box(r)),
                    ])
                    .unwrap();
                } else {
                    ir.call(&mut [
                        PcuHostArgument::read(inputs[0], black_box(a)),
                        PcuHostArgument::read(inputs[1], black_box(b)),
                        PcuHostArgument::read_write(outputs[0], black_box(q)),
                        PcuHostArgument::read_write(outputs[1], black_box(r)),
                    ])
                    .unwrap();
                }
            }
            2 => native
                .call_input_roles(
                    &[black_box(&a[..counts[0]]), black_box(&b[..counts[1]])][..input_count],
                    &counts[..input_count],
                    slots,
                    [black_box(q), black_box(r)],
                )
                .unwrap(),
            _ => ordinary::<T, N>(
                profile,
                black_box(a),
                black_box(b),
                black_box(q),
                black_box(r),
            ),
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
        "mlx_checked_div_rem_roles_{:?}_profile{profile}_host_joint_publish_{}",
        T::TYPE,
        if cfg!(feature = "portable-joint-control") {
            "portable_v1"
        } else {
            "unspecified"
        }
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
            assert_eq!(scored, 0);
            eprintln!(
                "Rust allocation census/div_rem_roles/{:?}/profile{profile}/{N}/{name}/{}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, invocation_score_callbacks={scored} (actual unique staging, both private reads, joint host publish; native/device allocations unknown)",
                T::TYPE,
                if cfg!(feature = "portable-joint-control") {
                    "portable_v1"
                } else {
                    "unspecified"
                },
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
fn profiles<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MlxSession) {
    let backend = session.checked_div_rem_role_backend();
    let mut p = source::repeated_prepare::<T, N, _>(&backend).unwrap();
    compare::<T, N, _>(criterion, session, 0, |a, _b, q, r| p(q, a, r));
    let mut p = source::unread_prepare::<T, N, _>(&backend).unwrap();
    compare::<T, N, _>(criterion, session, 1, |a, _b, q, r| p(&[], r, a, q));
    let mut p = source::reordered_prepare::<T, N, _>(&backend).unwrap();
    compare::<T, N, _>(criterion, session, 2, |a, b, q, r| p(b, r, a, q));
    let mut p = source::grid_prepare::<T, N, _>(&backend).unwrap();
    compare::<T, N, _>(criterion, session, 3, |a, _b, q, r| p(&[], r, a, q));
    let mut p = source::scalar_divisor_prepare::<T, N, _>(&backend).unwrap();
    compare::<T, N, _>(criterion, session, 4, |a, b, q, r| p(q, &b[0], r, a));
    let mut p = source::scalar_grid_prepare::<T, N, _>(&backend).unwrap();
    compare::<T, N, _>(criterion, session, 5, |a, _b, q, r| p(&a[0], q, r));
}
pub fn run(criterion: &mut Criterion) {
    if !std::env::args().any(|arg| arg == "--test") {
        activity::guard();
    }
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    macro_rules! run{($($ty:ty),+)=>{$(profiles::<$ty,65>(criterion,&session);profiles::<$ty,4096>(criterion,&session);)+};}
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

//! Fresh host inputs/private completed integer output/status/host readback, all four peers.
#[rustfmt::skip]
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
use std::hint::black_box;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxSession,
    MlxRuntime,
    MlxHostKernelError,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedInteger,
    PcuScalar,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuDispatchIntegerBinaryOp as Op,
    PcuRangePolicy as Range,
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
#[path = "../../../tests/checked_integer/graph/graph.rs"]
mod graph;
#[path = "../source/source.rs"]
mod source;
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    1
}
trait Sample: PcuCheckedInteger {
    fn small(value: u8) -> Self;
}
macro_rules! native {($($ty:ty),+)=>{$(impl Sample for $ty {fn small(value:u8)->Self {Self::try_from(value).unwrap()}})+};}
native!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);
macro_rules! wide {
    ($ty:ty,$limbs:literal) => {
        impl Sample for $ty {
            fn small(value: u8) -> Self {
                let mut limbs = [0; $limbs];
                limbs[0] = u64::from(value);
                Self::from_limbs_le(limbs)
            }
        }
    };
}
wide!(PcuI256, 4);
wide!(PcuU256, 4);
wide!(PcuI512, 8);
wide!(PcuU512, 8);
fn verify<T: PcuScalar>(output: &[T], wanted: &[T]) {
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), output).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), wanted).bytes()
    );
}
#[allow(clippy::too_many_lines)] // Four fixed concrete physical peers share frozen changing banks and separate caller census.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Same mutable Criterion entry; untimed census uses no Criterion sampling.
fn peers<
    T: Sample,
    const N: usize,
    F: FnMut(&[T], &[T], &mut [T]) -> Result<(), MlxHostKernelError>,
>(
    criterion: &mut Criterion,
    session: &MlxSession,
    op: Op,
    range: Range,
    mut source: F,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        numerical_mode: pcu_facade::PcuNumericalMode::Strict,
        range_policy: range,
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    let backend = session.checked_integer_backend();
    let mut graph = graph::fixture_profile::<T, _>(
        u32::try_from(N).unwrap(),
        op,
        range,
        false,
        [false; 2],
        |ir| backend.prepare_host_kernel(ir),
    )
    .unwrap();
    let mut native = session
        .prepare_checked_integer_control(T::TYPE, op, range, N, [N; 2], [false; 2])
        .unwrap();
    let sentinel = T::small(7);
    let banks: [(Vec<T>, Vec<T>, Vec<T>); 64] = std::array::from_fn(|bank| {
        let left: Vec<T> = (0..N)
            .map(|lane| T::small(u8::try_from((lane ^ bank) % 3 + 3).unwrap()))
            .collect();
        let right: Vec<T> = (0..N)
            .map(|lane| T::small(u8::try_from((lane ^ bank) % 2 + 1).unwrap()))
            .collect();
        let mut wanted: Vec<T> = left
            .iter()
            .zip(&right)
            .map(|(&a, &b)| {
                match op {
                    Op::Add => a.pcu_checked_add(b),
                    Op::Sub => a.pcu_checked_sub(b),
                    Op::Mul => a.pcu_checked_mul(b),
                }
                .unwrap()
            })
            .collect();
        wanted.extend([sentinel; 2]);
        (left, right, wanted)
    });
    let mut output = vec![sentinel; N + 2];
    let mut execute = |route, bank: usize, output: &mut [T]| {
        let (left, right, _) = &banks[bank];
        match route {
            0 => source(black_box(left), black_box(right), black_box(output)).unwrap(),
            1 => graph
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(4, 1), black_box(output)),
                    PcuHostArgument::read(PcuBindingRef::new(3, 2), black_box(right)),
                    PcuHostArgument::read(PcuBindingRef::new(2, 3), black_box(left)),
                ])
                .unwrap(),
            2 => native
                .call([black_box(left), black_box(right)], black_box(output))
                .unwrap(),
            3 => match (op, range) {
                (Op::Add, Range::Reject) => {
                    source::add::<T, N>(black_box(left), black_box(right), black_box(output))
                }
                (Op::Sub, Range::Reject) => {
                    source::sub::<T, N>(black_box(left), black_box(right), black_box(output))
                }
                (Op::Mul, Range::Reject) => {
                    source::mul::<T, N>(black_box(left), black_box(right), black_box(output))
                }
                (Op::Add, Range::Clamp) => {
                    source::add_clamp::<T, N>(black_box(left), black_box(right), black_box(output))
                }
                (Op::Sub, Range::Clamp) => {
                    source::sub_clamp::<T, N>(black_box(left), black_box(right), black_box(output))
                }
                (Op::Mul, Range::Clamp) => {
                    source::mul_clamp::<T, N>(black_box(left), black_box(right), black_box(output))
                }
            }
            .unwrap(),
            _ => unreachable!(),
        }
    };
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    for (route, name) in [
        "source_prepared",
        "explicit_graph",
        "direct_native",
        "ordinary_pcu",
    ]
    .into_iter()
    .enumerate()
    {
        for (bank, (_, _, wanted)) in banks.iter().enumerate() {
            execute(route, bank, &mut output);
            verify(&output, wanted);
        }
        #[cfg(feature = "allocation-census")]
        {
            let before = SCORES.load(std::sync::atomic::Ordering::Relaxed);
            let mut total = census::Census::default();
            for (bank, (_, _, wanted)) in banks.iter().enumerate() {
                let ((), count) = census::measure(|| execute(route, bank, &mut output));
                total.alloc_calls += count.alloc_calls;
                total.realloc_calls += count.realloc_calls;
                total.dealloc_calls += count.dealloc_calls;
                total.requested_bytes += count.requested_bytes;
                verify(&output, wanted);
            }
            let scored = SCORES.load(std::sync::atomic::Ordering::Relaxed) - before;
            assert_eq!(scored, 0, "warm scalar call rescored candidates");
            eprintln!(
                "Rust allocation census/integer/{:?}/{op:?}/{range:?}/{N}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, invocation_score_callbacks={scored} (64 changing prebuilt banks; private GPU/status/completion/readback included; native/device allocations unknown)",
                T::TYPE,
                total.alloc_calls,
                total.realloc_calls,
                total.dealloc_calls,
                total.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        criterion.bench_with_input(
            BenchmarkId::new(
                format!("mlx_integer_{:?}_{op:?}_{range:?}_{N}", T::TYPE),
                name,
            ),
            &route,
            |bench, &route| {
                let mut bank = 0;
                bench.iter(|| {
                    bank = (bank + 1) % 64;
                    execute(route, bank, &mut output);
                    black_box(&output);
                });
            },
        );
        for (bank, (_, _, wanted)) in banks.iter().enumerate() {
            execute(route, bank, &mut output);
            verify(&output, wanted);
        }
    }
}
fn format<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MlxSession) {
    let backend = session.checked_integer_backend();
    peers::<T, N, _>(
        criterion,
        session,
        Op::Add,
        Range::Reject,
        source::add_prepare::<T, N, _>(&backend).unwrap(),
    );
    peers::<T, N, _>(
        criterion,
        session,
        Op::Sub,
        Range::Reject,
        source::sub_prepare::<T, N, _>(&backend).unwrap(),
    );
    peers::<T, N, _>(
        criterion,
        session,
        Op::Mul,
        Range::Reject,
        source::mul_prepare::<T, N, _>(&backend).unwrap(),
    );
    peers::<T, N, _>(
        criterion,
        session,
        Op::Add,
        Range::Clamp,
        source::add_clamp_prepare::<T, N, _>(&backend).unwrap(),
    );
    peers::<T, N, _>(
        criterion,
        session,
        Op::Sub,
        Range::Clamp,
        source::sub_clamp_prepare::<T, N, _>(&backend).unwrap(),
    );
    peers::<T, N, _>(
        criterion,
        session,
        Op::Mul,
        Range::Clamp,
        source::mul_clamp_prepare::<T, N, _>(&backend).unwrap(),
    );
}
fn type_profile<T: Sample>(criterion: &mut Criterion, session: &MlxSession) {
    format::<T, 65>(criterion, session);
    format::<T, 4096>(criterion, session);
}
pub fn run(criterion: &mut Criterion) {
    activity();
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    type_profile::<u8>(criterion, &session);
    type_profile::<i8>(criterion, &session);
    type_profile::<u16>(criterion, &session);
    type_profile::<i16>(criterion, &session);
    type_profile::<u32>(criterion, &session);
    type_profile::<i32>(criterion, &session);
    type_profile::<u64>(criterion, &session);
    type_profile::<i64>(criterion, &session);
    type_profile::<u128>(criterion, &session);
    type_profile::<i128>(criterion, &session);
    type_profile::<PcuU256>(criterion, &session);
    type_profile::<PcuI256>(criterion, &session);
    type_profile::<PcuU512>(criterion, &session);
    type_profile::<PcuI512>(criterion, &session);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    activity();
}

fn activity() {
    // Correctness/caller census records load; statistical timing still requires idle hardware.
    if cfg!(feature = "allocation-census") || std::env::args().any(|argument| argument == "--test")
    {
        let output = std::process::Command::new("/usr/sbin/ioreg")
            .args(["-r", "-c", "AGXAccelerator", "-l"])
            .output()
            .unwrap();
        assert!(output.status.success());
        let text = String::from_utf8(output.stdout).unwrap();
        let value: u32 = text
            .split("\"Device Utilization %\"=")
            .nth(1)
            .unwrap()
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap();
        println!("MLX integer correctness-only observed GPU {value}% (no idle/timing claim)");
    } else {
        activity::guard();
    }
}

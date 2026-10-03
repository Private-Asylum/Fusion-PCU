//! Four matched unary full-capacity source/prepared/native boundaries.
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
    PcuBf16Bits,
    PcuDispatchFloatUnaryOp as Op,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy as Policy,
    PcuRangePolicy as Range,
};
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[path = "captured.rs"]
mod captured;
#[cfg(feature = "allocation-census")]
#[path = "../../tensor_binary/census/census.rs"]
mod census;
#[path = "../../../tests/checked_unary/graph/graph.rs"]
mod graph;
#[path = "oracle.rs"]
mod oracle;
#[path = "ordinary.rs"]
mod ordinary;
use super::source;
use oracle::Sample;
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    1
}
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
// Each exact tuple preserves four matched physical boundaries and 64 independent changed-bank checks.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Criterion registration context is shared with untimed census mode.
fn case<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    profile: usize,
) {
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    // Immutable owned input construction uses Reject; invocation-local flags retain the exact
    // range/UF request without admitting owned graph Clamp.
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    let invoke = ordinary::select::<T, N>(op, policy, range, profile);
    let full = N.checked_add(6).unwrap();
    let broadcast = profile == 2;
    let mut prepared = captured::prepare::<T, N>(session, op, policy, range, profile, full);
    let prepare = |ir: &pcu_facade::PcuDispatchKernelIr<'_>| {
        let request = *ir;

        session.prepare_unary_host_kernel_with_input_extents(&request, &[full])
    };
    let mut explicit = if range == Range::Reject && !broadcast {
        graph::fixture::<T, _>(u32::try_from(N).unwrap(), op, policy, profile == 1, prepare)
    } else {
        graph::fixture_profile::<T, _>(
            u32::try_from(N).unwrap(),
            op,
            policy,
            range,
            profile == 1,
            broadcast,
            prepare,
        )
    }
    .unwrap();
    let mut native = session
        .prepare_checked_unary_control_with_input_extent(
            T::TYPE,
            op,
            policy,
            range,
            N,
            broadcast,
            full,
        )
        .unwrap();
    // Actual input uploads and full-capacity synthetic priming are cold. Allocation counts below
    // observe the caller Rust path, not MLX/device heaps or SDK-internal compilation.
    let hosts: Vec<Vec<T>> = (0..64).map(|bank| oracle::bank(N, full, bank)).collect();
    // Every actual read prefix is distinct, including scalar broadcast lane0. No host float
    // arithmetic or low-format conversion determines these finite normal encoding banks.
    for (bank, input) in hosts.iter().enumerate() {
        for earlier in &hosts[..bank] {
            assert_ne!(input[0].bits(), earlier[0].bits());
        }
    }
    let inputs: Vec<_> = hosts
        .iter()
        .map(|bank| session.upload_encoded(bank).unwrap())
        .collect();
    let ordinary_inputs: Vec<_> = hosts
        .iter()
        .map(|bank| source::retain(bank).unwrap())
        .collect();
    let expected: Vec<Vec<T>> = hosts
        .iter()
        .map(|bank| oracle::expected(bank, N, op, broadcast))
        .collect();
    let sentinel = T::raw(17);
    let mut output = vec![sentinel; N + 3];
    let mut original = vec![sentinel; full + 3];
    let mut execute = |route: usize, bank: usize, destination: &mut [T]| {
        if route == 3 {
            invoke(
                &[],
                black_box(destination),
                &hosts[bank][0],
                black_box(&ordinary_inputs[bank]),
            )
            .unwrap();
            return;
        }
        let completion = match route {
            0 => prepared.execute_resident(black_box(&inputs[bank])),
            1 => explicit.execute_resident(black_box(&inputs[bank])),
            2 => native.execute_resident(black_box(&inputs[bank])),
            _ => unreachable!("four matched unary prefix peers"),
        }
        .unwrap();
        let (completed, notice) = completion.into_parts();
        assert!(
            notice.is_none(),
            "finite normal/signed-zero bank must not recover"
        );
        completed.read_into(black_box(destination)).unwrap();
        drop(completed);
    };
    let verify = |bank: usize, destination: &[T]| {
        oracle::compare(&destination[..N], &expected[bank]);
        oracle::compare(&destination[N..], &[sentinel; 3]);
    };
    let profile_name = ["dense", "grid", "scalar-broadcast"][profile];
    let cohort = "normal_unary_roles";
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "mlx_{cohort}/{:?}/{op:?}/{policy:?}/{range:?}/{profile_name}",
        T::TYPE
    ));
    for (route, name) in [
        "captured-source",
        "explicit-ir",
        "direct-native",
        "ordinary-pcu",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in [0, 23, 63] {
            output.fill(sentinel);
            execute(route, bank, &mut output);
            verify(bank, &output);
            inputs[bank].read_into(&mut original).unwrap();
            oracle::compare(&original[..full], &hosts[bank]);
            ordinary_inputs[bank].read_into(&mut original).unwrap();
            oracle::compare(&original[..full], &hosts[bank]);
            oracle::compare(&original[full..], &[sentinel; 3]);
        }
        #[cfg(feature = "allocation-census")]
        {
            let callbacks_before = SCORES.load(std::sync::atomic::Ordering::Relaxed);
            let mut total = census::Census::default();
            for bank in 0..64 {
                output.fill(sentinel);
                let ((), count) = census::measure(|| execute(route, bank, &mut output));
                total.alloc_calls += count.alloc_calls;
                total.realloc_calls += count.realloc_calls;
                total.dealloc_calls += count.dealloc_calls;
                total.requested_bytes += count.requested_bytes;
                verify(bank, &output);
            }
            let scored = SCORES.load(std::sync::atomic::Ordering::Relaxed) - callbacks_before;
            assert_eq!(scored, 0, "warm ordinary unary prefix must not rescore");
            eprintln!(
                "Rust allocation census/{cohort}/{:?}/{op:?}/{policy:?}/{range:?}/{profile_name}/{N}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, invocation_score_callbacks={scored} (retained full input, private output, terminal host-prefix read/drop; SDK heap/JIT unknown)",
                T::TYPE,
                total.alloc_calls,
                total.realloc_calls,
                total.dealloc_calls,
                total.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        group.bench_with_input(BenchmarkId::new(name, N), &N, |bench, _| {
            bench.iter(|| execute(route, 23, &mut output));
        });
        for bank in [0, 23, 63] {
            output.fill(sentinel);
            execute(route, bank, &mut output);
            verify(bank, &output);
            inputs[bank].read_into(&mut original).unwrap();
            oracle::compare(&original[..full], &hosts[bank]);
            ordinary_inputs[bank].read_into(&mut original).unwrap();
            oracle::compare(&original[..full], &hosts[bank]);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
}
fn format<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MlxSession) {
    for op in [Op::Neg, Op::Relu] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::AllowGradualUnderflow,
            Policy::RejectSubnormalResult,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                for profile in 0..3 {
                    case::<T, N>(criterion, session, op, policy, range, profile);
                }
            }
        }
    }
}
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    macro_rules! formats {
        ($count:literal) => {
            format::<PcuF16Bits, $count>(criterion, &session);
            format::<PcuBf16Bits, $count>(criterion, &session);
            format::<PcuF8E4M3FnBits, $count>(criterion, &session);
            format::<PcuF8E5M2Bits, $count>(criterion, &session);
            format::<f32, $count>(criterion, &session);
            format::<f64, $count>(criterion, &session);
        };
    }
    formats!(65);
    formats!(4096);
}

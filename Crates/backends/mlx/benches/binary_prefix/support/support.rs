//! Four matched binary full-capacity source/prepared/native boundaries.
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
    PcuDispatchFloatBinaryOp as Op,
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
#[path = "../../../tests/checked_binary/graph/graph.rs"]
mod graph;
#[path = "oracle.rs"]
mod oracle;
#[path = "ordinary.rs"]
mod ordinary;
#[path = "../source/source.rs"]
mod source;
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
    // Graph owners are created under Reject; actual invocation flags retain UF/range locally.
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    let invoke = ordinary::select::<T, N>(op, policy, range, profile);
    let full = [N.checked_add(6).unwrap(), N.checked_add(9).unwrap()];
    let repeated = profile == 2;
    let unique_count = if repeated { 1 } else { 2 };
    let broadcast = [repeated, false];
    let mut prepared =
        captured::prepare::<T, N>(session, op, policy, range, profile, &full[..unique_count]);
    let prepare = |ir: &pcu_facade::PcuDispatchKernelIr<'_>| {
        session
            .checked_binary_backend()
            .prepare_host_kernel_with_input_extents(ir, &full[..unique_count])
    };
    let mut explicit = if repeated {
        graph::fixture_roles::<T, _>(
            u32::try_from(N).unwrap(),
            op,
            policy,
            range,
            false,
            broadcast,
            true,
            prepare,
        )
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
    let minimum = [N, N]; // Repeated zero+indexed loads share one N-element actual read.
    let mut native = session
        .prepare_checked_binary_control_with_input_extents(
            T::TYPE,
            op,
            policy,
            range,
            N,
            minimum,
            broadcast,
            [full[0], if repeated { full[0] } else { full[1] }],
        )
        .unwrap();
    // Actual input uploads and full-capacity synthetic priming are cold. Allocation counts below
    // observe the caller Rust path, not MLX/device heaps or SDK-internal compilation.
    let hosts: Vec<Vec<T>> = (0..64)
        .map(|bank| oracle::bank(N, full[0], bank, false))
        .collect();
    let right_hosts: Vec<Vec<T>> = (0..64)
        .map(|bank| oracle::bank(N, full[1], bank, true))
        .collect();
    let inputs: Vec<_> = hosts
        .iter()
        .map(|bank| session.upload_encoded(bank).unwrap())
        .collect();
    let rights: Vec<_> = if repeated {
        Vec::new()
    } else {
        right_hosts
            .iter()
            .map(|bank| session.upload_encoded(bank).unwrap())
            .collect()
    };
    let ordinary_inputs: Vec<_> = hosts
        .iter()
        .map(|bank| source::retain(bank).unwrap())
        .collect();
    let ordinary_rights: Vec<_> = if repeated {
        Vec::new()
    } else {
        right_hosts
            .iter()
            .map(|bank| source::retain(bank).unwrap())
            .collect()
    };
    let expected: Vec<Vec<T>> = hosts
        .iter()
        .zip(&right_hosts)
        .map(|(a, b)| oracle::expected(a, b, N, op, repeated))
        .collect();
    let sentinel = T::value(17.0);
    let mut output = vec![sentinel; N + 3];
    let mut original = vec![sentinel; full[0] + 3];
    let mut original_right = vec![sentinel; full[1] + 3];
    let mut execute = |route: usize, bank: usize, destination: &mut [T]| {
        if route == 3 {
            let right = if repeated {
                &ordinary_inputs[bank]
            } else {
                &ordinary_rights[bank]
            };
            invoke(
                black_box(&ordinary_inputs[bank]),
                black_box(right),
                black_box(destination),
            )
            .unwrap();
            return;
        }
        let actual = [
            &inputs[bank],
            if repeated {
                &inputs[bank]
            } else {
                &rights[bank]
            },
        ];
        let completion = match route {
            0 => prepared.execute_resident(black_box(&actual[..unique_count])),
            1 => explicit.execute_resident(black_box(&actual[..unique_count])),
            2 => native.execute_resident(black_box(actual)),
            _ => unreachable!("four matched binary prefix peers"),
        }
        .unwrap();
        let (completed, notice) = completion.into_parts();
        assert!(
            notice.is_none(),
            "finite normal dyadic bank must not recover"
        );
        completed.read_into(black_box(destination)).unwrap();
        drop(completed);
    };
    let verify = |bank: usize, destination: &[T]| {
        oracle::compare(&destination[..N], &expected[bank]);
        oracle::compare(&destination[N..], &[sentinel; 3]);
    };
    let profile_name = ["dense", "grid", "repeated-zero-indexed"][profile];
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "mlx_binary_prefix/{:?}/{op:?}/{policy:?}/{range:?}/{profile_name}",
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
            oracle::compare(&original[..full[0]], &hosts[bank]);
            ordinary_inputs[bank].read_into(&mut original).unwrap();
            oracle::compare(&original[..full[0]], &hosts[bank]);
            if !repeated {
                rights[bank].read_into(&mut original_right).unwrap();
                oracle::compare(&original_right[..full[1]], &right_hosts[bank]);
                ordinary_rights[bank]
                    .read_into(&mut original_right)
                    .unwrap();
                oracle::compare(&original_right[..full[1]], &right_hosts[bank]);
                oracle::compare(&original_right[full[1]..], &[sentinel; 3]);
            }
            oracle::compare(&original[full[0]..], &[sentinel; 3]);
        }
        #[cfg(feature = "allocation-census")]
        {
            #[cfg(feature = "binary-census")]
            fusion_pcu_mlx::reset_binary_call_census();
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
            assert_eq!(scored, 0, "warm ordinary binary prefix must not rescore");
            eprintln!(
                "Rust allocation census/binary_prefix/{:?}/{op:?}/{policy:?}/{range:?}/{profile_name}/{N}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, invocation_score_callbacks={scored} (retained full input, private output, terminal host-prefix read/drop; SDK heap/JIT unknown)",
                T::TYPE,
                total.alloc_calls,
                total.realloc_calls,
                total.dealloc_calls,
                total.requested_bytes
            );
            #[cfg(feature = "binary-census")]
            {
                let adapter = fusion_pcu_mlx::binary_call_census();
                assert_eq!(adapter.exact_constructor_calls, 0);
                assert_eq!(adapter.prefix_constructor_calls, 0);
                assert_eq!(adapter.prime_calls, 0);
                assert_eq!(adapter.table_symbol_attempts, 0);
                assert_eq!(adapter.apply_calls, 64);
                eprintln!(
                    "Binary adapter census/binary_prefix/{:?}/{op:?}/{policy:?}/{range:?}/{profile_name}/{N}/{name}: calls=64 exact_constructors={} prefix_constructors={} primes={} applies={} table_symbol_attempts={} (adapter attempts only; SDK-internal JIT and allocation unknown)",
                    T::TYPE,
                    adapter.exact_constructor_calls,
                    adapter.prefix_constructor_calls,
                    adapter.prime_calls,
                    adapter.apply_calls,
                    adapter.table_symbol_attempts
                );
            }
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
            oracle::compare(&original[..full[0]], &hosts[bank]);
            ordinary_inputs[bank].read_into(&mut original).unwrap();
            oracle::compare(&original[..full[0]], &hosts[bank]);
            if !repeated {
                rights[bank].read_into(&mut original_right).unwrap();
                oracle::compare(&original_right[..full[1]], &right_hosts[bank]);
                ordinary_rights[bank]
                    .read_into(&mut original_right)
                    .unwrap();
                oracle::compare(&original_right[..full[1]], &right_hosts[bank]);
                oracle::compare(&original_right[full[1]..], &[sentinel; 3]);
            }
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
}
fn format<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MlxSession) {
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
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

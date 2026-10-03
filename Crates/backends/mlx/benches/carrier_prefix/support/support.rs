//! Four equal full-resident-input to fresh private output/read/drop boundaries.
#[rustfmt::skip]
use std::hint::black_box;
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxEncodedArray,
    MlxPreparedCarrierHostKernel,
    MlxRuntime,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBf16Bits,
    PcuDispatchKernelIr,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuI512,
    PcuTensor,
    PcuU256,
    PcuU512,
};
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../tensor_binary/census/census.rs"]
mod census;
#[path = "../../../tests/carrier_map/graph/graph.rs"]
mod graph;
#[path = "../../../tests/encoded_carrier/support/support.rs"]
mod oracle;
#[path = "../source/source.rs"]
mod source;
#[rustfmt::skip]
use oracle::{
    compare,
    Sample,
};
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    1
}
struct Bank<T: Sample> {
    values: Vec<T>,
    native: MlxEncodedArray,
    ordinary: PcuTensor<T>,
}
fn captured<T: Sample, const N: usize>(
    session: &MlxSession,
    profile: usize,
    full: usize,
) -> MlxPreparedCarrierHostKernel {
    let prepare = |ir: &PcuDispatchKernelIr<'_>| {
        session.prepare_carrier_host_kernel_with_input_extents(ir, &[full])
    };
    match profile {
        0 => source::copy_ir::<T, N>(&source::copy_bindings::<T>())
            .unwrap()
            .with_ir(prepare),
        1 => source::grid_ir::<T, N>(&source::grid_bindings::<T>())
            .unwrap()
            .with_ir(prepare),
        2 => source::broadcast_ir::<T, N>(&source::broadcast_bindings::<T>())
            .unwrap()
            .with_ir(prepare),
        _ => unreachable!("three exact carrier source profiles"),
    }
    .unwrap()
}
fn ordinary<T: Sample, const N: usize>(profile: usize, input: &PcuTensor<T>, output: &mut [T]) {
    match profile {
        0 => source::copy::<T, N>(input, output),
        1 => source::grid::<T, N>(input, output),
        2 => source::broadcast::<T, N>(input, output),
        _ => unreachable!("three exact carrier source profiles"),
    }
    .unwrap();
}
#[allow(clippy::too_many_lines)] // All four physical peer boundaries and each changing-call oracle remain visible together.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Criterion registration ABI shared by untimed census mode.
fn format<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MlxSession) {
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    let full = N.checked_add(6).unwrap();
    // All input uploads, truthful full-size native priming and global source preparation are
    // cold. SDK/device heap and internal compiler work are not observed by the Rust allocator.
    let banks: Vec<Bank<T>> = (0..64)
        .map(|bank| {
            let values: Vec<T> = (0..full)
                .map(|lane| T::sample(u8::try_from((lane + bank * 23) % 256).unwrap()))
                .collect();
            Bank {
                native: session.upload_encoded(&values).unwrap(),
                ordinary: source::retain(&values).unwrap(),
                values,
            }
        })
        .collect();
    let sentinel = T::sample(91);
    let mut output = vec![sentinel; N + 3];
    let mut original = vec![sentinel; full + 3];
    let verify_input = |bank: usize, scratch: &mut [T]| {
        scratch.fill(sentinel);
        banks[bank].native.read_into(scratch).unwrap();
        compare(&scratch[..full], &banks[bank].values);
        compare(&scratch[full..], &[sentinel; 3]);
        scratch.fill(sentinel);
        banks[bank].ordinary.read_into(scratch).unwrap();
        compare(&scratch[..full], &banks[bank].values);
        compare(&scratch[full..], &[sentinel; 3]);
    };
    for (profile, profile_name) in ["dense", "grid", "scalar-broadcast"]
        .into_iter()
        .enumerate()
    {
        let mut prepared = captured::<T, N>(session, profile, full);
        let mut explicit = graph::fixture::<T, _>(
            u32::try_from(N).unwrap(),
            profile == 2,
            profile == 1,
            |ir| session.prepare_carrier_host_kernel_with_input_extents(ir, &[full]),
        )
        .unwrap();
        let mut native = session
            .prepare_carrier_control_with_input_extent(T::TYPE, N, profile == 2, full)
            .unwrap();
        let mut execute = |route: usize, bank: usize, destination: &mut [T]| {
            let input = &banks[bank];
            if route == 3 {
                ordinary::<T, N>(profile, black_box(&input.ordinary), black_box(destination));
            } else {
                let completed = match route {
                    0 => prepared.execute_resident(black_box(&input.native)),
                    1 => explicit.execute_resident(black_box(&input.native)),
                    2 => native.execute_resident(black_box(&input.native)),
                    _ => unreachable!("four matched carrier-prefix peers"),
                }
                .unwrap()
                .into_parts()
                .0;
                completed.read_into(black_box(destination)).unwrap();
                drop(completed);
            }
        };
        let verify = |bank: usize, destination: &[T]| {
            let input = &banks[bank].values;
            if profile == 2 {
                for value in &destination[..N] {
                    compare(std::slice::from_ref(value), &input[..1]);
                }
            } else {
                compare(&destination[..N], &input[..N]);
            }
            compare(&destination[N..], &[sentinel; 3]);
        };
        #[cfg(not(feature = "allocation-census"))]
        let mut group =
            criterion.benchmark_group(format!("mlx_carrier_prefix/{:?}/{profile_name}", T::TYPE));
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
                verify_input(bank, &mut original);
                output.fill(sentinel);
                execute(route, bank, &mut output);
                verify(bank, &output);
            }
            #[cfg(feature = "allocation-census")]
            {
                let mut total = census::Census::default();
                let scored = SCORES.load(std::sync::atomic::Ordering::Relaxed);
                #[cfg(feature = "carrier-census")]
                fusion_pcu_mlx::reset_carrier_call_census();
                for bank in 0..64 {
                    output.fill(sentinel);
                    let ((), count) = census::measure(|| execute(route, bank, &mut output));
                    total.alloc_calls += count.alloc_calls;
                    total.realloc_calls += count.realloc_calls;
                    total.dealloc_calls += count.dealloc_calls;
                    total.requested_bytes += count.requested_bytes;
                    verify(bank, &output);
                }
                let scored = SCORES.load(std::sync::atomic::Ordering::Relaxed) - scored;
                assert_eq!(scored, 0, "warm full-shape carrier calls must not rescore");
                #[cfg(feature = "carrier-census")]
                {
                    let adapter = fusion_pcu_mlx::carrier_call_census();
                    assert_eq!(adapter.exact_constructor_calls, 0);
                    assert_eq!(adapter.prefix_constructor_calls, 0);
                    assert_eq!(adapter.prime_calls, 0);
                    assert_eq!(adapter.table_symbol_attempts, 0);
                    assert_eq!(adapter.apply_calls, 64);
                    eprintln!(
                        "Carrier adapter census/carrier_prefix/{:?}/{profile_name}/{N}/{name}: calls=64 exact_constructors={} prefix_constructors={} primes={} applies={} table_symbol_attempts={} (adapter attempts only; SDK-internal JIT and allocation unknown)",
                        T::TYPE,
                        adapter.exact_constructor_calls,
                        adapter.prefix_constructor_calls,
                        adapter.prime_calls,
                        adapter.apply_calls,
                        adapter.table_symbol_attempts
                    );
                }
                eprintln!(
                    "Rust allocation census/carrier_prefix/{:?}/{profile_name}/{N}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, invocation_score_callbacks={scored} (retained full input, fresh private output, terminal host-prefix read/drop; native/device allocations unknown)",
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
                verify_input(bank, &mut original);
            }
        }
        #[cfg(not(feature = "allocation-census"))]
        group.finish();
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    macro_rules! all {($n:expr;$($ty:ty),+)=>{$(format::<$ty,$n>(criterion,&session);)+};}
    all!(65;u8,i8,u16,i16,u32,i32,u64,i64,u128,i128,PcuU256,PcuI256,PcuU512,PcuI512,
        PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,f32,f64,PcuF128Bits,PcuF256Bits);
    all!(4096;u8,i8,u16,i16,u32,i32,u64,i64,u128,i128,PcuU256,PcuI256,PcuU512,PcuI512,
        PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,f32,f64,PcuF128Bits,PcuF256Bits);
}

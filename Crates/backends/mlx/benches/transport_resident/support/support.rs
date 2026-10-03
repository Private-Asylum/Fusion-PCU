//! Cold banks and full shapes; separate 64-changing-call caller census, no native heap claims.
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxEncodedArray,
    MlxNativeTransportWorkload,
    MlxRuntime,
    MlxTransportInput,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuScalar,
    PcuTensor,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../transport/census/census.rs"]
mod census;
#[path = "../../transport/graph/graph.rs"]
mod graph;
#[path = "../../../tests/encoded_carrier/support/support.rs"]
mod oracle;
#[path = "../publication/publication.rs"]
mod publication;
#[path = "../source/source.rs"]
mod source;
#[rustfmt::skip]
use oracle::{
    Sample,
    compare,
};
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    1
}
struct Bank<T: PcuScalar> {
    input: Vec<T>,
    seed: Vec<T>,
    native: [MlxEncodedArray; 2],
    ordinary: [PcuTensor<T>; 2],
}
fn bytes<T: PcuScalar>(data: &[T]) -> Vec<u8> {
    pcu_facade::PcuHostArgument::read(pcu_facade::PcuBindingRef::new(0, 0), data)
        .bytes()
        .to_vec()
}
fn verify_inputs<T: Sample>(banks: &[Bank<T>], scratch: &mut [T], sentinel: T) {
    for bank in banks {
        for slot in 0..2 {
            scratch.fill(sentinel);
            bank.native[slot].read_into(scratch).unwrap();
            let expected = if slot == 0 { &bank.input } else { &bank.seed };
            compare(&scratch[..expected.len()], expected);
            compare(
                &scratch[expected.len()..],
                &vec![sentinel; scratch.len() - expected.len()],
            );
            scratch.fill(sentinel);
            bank.ordinary[slot].read_into(scratch).unwrap();
            compare(&scratch[..expected.len()], expected);
        }
    }
}
#[allow(clippy::too_many_lines)] // Keep four equal physical boundaries and separate per-call observers visible together.
fn geometry<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &fusion_pcu_mlx::MlxSession,
    grid: bool,
) {
    let full = N + 7;
    let banks: Vec<Bank<T>> = (0..64_u8)
        .map(|phase| {
            let input: Vec<T> = (0..full)
                .map(|lane| T::sample(u8::try_from(lane % 256).unwrap().wrapping_add(phase)))
                .collect();
            let seed: Vec<T> = (0..3_u8)
                .map(|lane| T::sample(lane.wrapping_add(phase)))
                .collect();
            let native = [
                session.upload_encoded(&input).unwrap(),
                session.upload_encoded(&seed).unwrap(),
            ];
            let ordinary = [
                source::retain(&input).unwrap(),
                source::retain(&seed).unwrap(),
            ];
            Bank {
                input,
                seed,
                native,
                ordinary,
            }
        })
        .collect();
    for (index, bank) in banks.iter().enumerate() {
        for prior in &banks[..index] {
            assert_ne!(bytes(&bank.input), bytes(&prior.input));
        }
    }
    let prepare = |ir: &pcu_facade::PcuDispatchKernelIr<'_>| {
        session
            .transport_host_backend()
            .prepare_host_kernel_with_input_extents(ir, &[full, 3])
    };
    let mut captured = if grid {
        source::saved_grid_ir::<T, N>(&source::saved_grid_bindings::<T>())
            .unwrap()
            .with_ir(prepare)
    } else {
        source::saved_ir::<T, N>(&source::saved_bindings::<T>())
            .unwrap()
            .with_ir(prepare)
    }
    .unwrap();
    let mut explicit = graph::visit(
        T::TYPE,
        u32::try_from(N).unwrap(),
        grid,
        pcu_facade::PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        prepare,
    )
    .unwrap();
    let native = session
        .prepare_native_transport_control(
            T::TYPE,
            N,
            MlxNativeTransportWorkload::SavedInput,
            &[full, 3],
        )
        .unwrap();
    let sentinel = T::sample(0x93);
    let mut publication = publication::Publication::new::<N>(session, sentinel);
    let mut observed: [Vec<T>; 2] = std::array::from_fn(|_| vec![sentinel; N + 8]);
    let mut scratch = vec![sentinel; full + 3];
    verify_inputs(&banks, &mut scratch, sentinel);
    for (boundary, label) in ["host-prefix", "exact-owner", "prefix-owner"]
        .into_iter()
        .enumerate()
    {
        for route in 0..4 {
            publication.reset::<N>(session, boundary, sentinel);
            let count = if boundary == 1 { N } else { N + 5 };
            let mut ordinary: [PcuTensor<T>; 2] =
                std::array::from_fn(|_| source::retain(&vec![sentinel; count]).unwrap());
            let mut execute = |bank: usize, run: bool, verify: bool| {
                let bank = &banks[bank];
                if run {
                    if route == 3 {
                        let [left, right] = &mut ordinary;
                        if boundary == 0 {
                            let [left, right] = &mut publication.host;
                            if grid {
                                source::saved_grid::<T, N>(
                                    &bank.ordinary[0],
                                    &bank.ordinary[1],
                                    &mut [],
                                    left,
                                    right,
                                )
                            } else {
                                source::saved::<T, N>(
                                    &bank.ordinary[0],
                                    &bank.ordinary[1],
                                    &mut [],
                                    left,
                                    right,
                                )
                            }
                            .unwrap();
                        } else if grid {
                            source::saved_grid::<T, N>(
                                &bank.ordinary[0],
                                &bank.ordinary[1],
                                &mut [],
                                left,
                                right,
                            )
                            .unwrap();
                        } else {
                            source::saved::<T, N>(
                                &bank.ordinary[0],
                                &bank.ordinary[1],
                                &mut [],
                                left,
                                right,
                            )
                            .unwrap();
                        }
                    } else {
                        let result = if route == 2 {
                            native.execute(&[&bank.native[0], &bank.native[1]]).unwrap()
                        } else {
                            let kernel = if route == 0 {
                                &mut captured
                            } else {
                                &mut explicit
                            };
                            let roles = kernel.plan().input_bindings();
                            let inputs = [
                                MlxTransportInput::Resident {
                                    target: roles[0],
                                    array: &bank.native[0],
                                },
                                MlxTransportInput::Resident {
                                    target: roles[1],
                                    array: &bank.native[1],
                                },
                            ];
                            let [Some(stage), Some(output)] =
                                kernel.execute_inputs(&inputs).unwrap().into_outputs()
                            else {
                                panic!("two private writers")
                            };
                            (stage, output)
                        };
                        publication.commit(result, boundary);
                    }
                }
                if verify {
                    if boundary == 0 {
                        compare(&publication.host[0][..N], &vec![bank.seed[0]; N]);
                        compare(&publication.host[1][..N], &bank.input[..N]);
                        for host in &publication.host {
                            compare(&host[N..], &[sentinel; 5]);
                        }
                    } else {
                        for slot in 0..2 {
                            observed[slot].fill(sentinel);
                            if route == 3 {
                                ordinary[slot].read_into(&mut observed[slot]).unwrap();
                            } else {
                                publication.owners[slot]
                                    .read_into(&mut observed[slot])
                                    .unwrap();
                            }
                            if slot == 0 {
                                compare(&observed[slot][..N], &vec![bank.seed[0]; N]);
                            } else {
                                compare(&observed[slot][..N], &bank.input[..N]);
                            }
                            compare(
                                &observed[slot][N..],
                                &vec![sentinel; observed[slot].len() - N],
                            );
                        }
                    }
                }
            };
            for bank in 0..64 {
                execute(bank, true, true);
            }
            register::<N>(criterion, T::TYPE, grid, label, route, &mut execute);
            for bank in 0..64 {
                execute(bank, true, true);
            }
        }
    }
    verify_inputs(&banks, &mut scratch, sentinel);
}

#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Shared Criterion registration is mutable only in the uninstrumented timing build.
fn register<const N: usize>(
    criterion: &mut Criterion,
    scalar: pcu_facade::PcuScalarType,
    grid: bool,
    boundary: &str,
    route: usize,
    execute: &mut impl FnMut(usize, bool, bool),
) {
    let geometry = if grid { "grid" } else { "direct" };
    let name = [
        "captured-source",
        "explicit-ir",
        "independent-native",
        "ordinary-pcu",
    ][route];
    #[cfg(feature = "allocation-census")]
    {
        let _ = criterion;
        let before = SCORES.load(std::sync::atomic::Ordering::Relaxed);
        let mut total = census::Census::default();
        for bank in 0..64 {
            let ((), counts) = census::measure(|| execute(bank, true, false));
            total.alloc_calls += counts.alloc_calls;
            total.realloc_calls += counts.realloc_calls;
            total.dealloc_calls += counts.dealloc_calls;
            total.requested_bytes += counts.requested_bytes;
            execute(bank, false, true);
        }
        let scores = SCORES.load(std::sync::atomic::Ordering::Relaxed) - before;
        println!(
            "Rust allocation census/mlx_resident_transport/{scalar:?}/{geometry}/{N}/{boundary}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, invocation_score_callbacks={scores} (caller Rust only; SDK heap/JIT unknown)",
            total.alloc_calls, total.realloc_calls, total.dealloc_calls, total.requested_bytes
        );
    }
    #[cfg(not(feature = "allocation-census"))]
    {
        let mut group = criterion.benchmark_group(format!(
            "mlx_resident_transport/{scalar:?}/{geometry}/{boundary}"
        ));
        let mut bank = 0;
        group.bench_function(BenchmarkId::new(name, N), |bench| {
            bench.iter(|| {
                bank = (bank + 1) % 64;
                execute(bank, true, false);
            });
        });
        group.finish();
    }
}
pub fn run(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: genuine MLX required");
        return;
    }
    activity::guard();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    macro_rules! run {($($ty:ty),+)=>{$(for grid in [false,true] {geometry::<$ty,65>(criterion,&session,grid);geometry::<$ty,4096>(criterion,&session,grid);})+};}
    run!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        PcuI256,
        PcuU256,
        PcuI512,
        PcuU512,
        f32,
        f64,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        PcuF128Bits,
        PcuF256Bits
    );
    global::clear_thread_cache().unwrap();
    activity::guard();
}

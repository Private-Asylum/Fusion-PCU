//! Raw changing input banks and exact saved-value/tail observers outside capture.

use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[rustfmt::skip]
use pcu_facade::{
    PcuScalar,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
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
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalPreparedTransportKernel,
    MetalSession,
    MetalTransportPlan,
};
#[path = "../../operand_roles/support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../portable_unary/census/census.rs"]
mod census;
#[path = "../graph/graph.rs"]
mod graph;
#[path = "../source/source.rs"]
mod source;

trait Sample: PcuScalar {
    fn raw(lane: usize, phase: u8) -> Self;
}
macro_rules! samples {($($ty:ty=>$width:literal),+)=>{$(
    impl Sample for $ty {
        fn raw(lane:usize,phase:u8)->Self {
            let mut bytes=[0_u8;$width];
            for (index,byte) in bytes.iter_mut().enumerate() {
                *byte=match lane%7 {
                    0=>phase,
                    1=>0xff,
                    2=>if index==$width-1 {0x80} else {0},
                    3=>if index==0 {1} else {0},
                    _=>u8::try_from((lane*13+index*37)%256).unwrap().wrapping_add(phase),
                };
            }
            Self::decode_le(bytes)
        }
    }
)+};}
samples!(i8=>1,u8=>1,i16=>2,u16=>2,i32=>4,u32=>4,i64=>8,u64=>8,i128=>16,u128=>16,PcuI256=>32,PcuU256=>32,PcuI512=>64,PcuU512=>64,f32=>4,f64=>8,PcuF16Bits=>2,PcuBf16Bits=>2,PcuF8E4M3FnBits=>1,PcuF8E5M2Bits=>1,PcuF128Bits=>16,PcuF256Bits=>32);
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(_: &pcu_facade::global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    1
}
fn bytes<T: PcuScalar>(values: &[T]) -> Vec<u8> {
    PcuHostArgument::read(PcuBindingRef::new(0, 0), values)
        .bytes()
        .to_vec()
}

struct Bank<T> {
    input: Vec<T>,
    seed: T,
    stage: Vec<u8>,
    output: Vec<u8>,
}
fn banks<T: Sample, const N: usize>() -> Vec<Bank<T>> {
    (0..64_u8)
        .map(|phase| {
            let input: Vec<T> = (0..N + 7).map(|lane| T::raw(lane, phase)).collect();
            let seed = T::raw(0, phase);
            let stage = [bytes(&[seed]).repeat(N), bytes(&[T::raw(9, 83); 5])].concat();
            let output = [bytes(&input[..N]), bytes(&[T::raw(11, 79); 3])].concat();
            Bank {
                input,
                seed,
                stage,
                output,
            }
        })
        .collect()
}

struct Native {
    kernel: MetalPreparedTransportKernel,
    stage: Vec<u8>,
    output: Vec<u8>,
}
impl Native {
    fn call<T: PcuScalar, const N: usize>(
        &mut self,
        session: &MetalSession,
        input: &[T],
        seed: &T,
        stage: &mut [T],
        output: &mut [T],
    ) {
        // Identical four fresh private banks and two initial prefix copies.
        let input_view = PcuHostArgument::read(graph::INPUT, &input[..N]);
        let seed_view = PcuHostArgument::read(graph::SEED, core::slice::from_ref(seed));
        let input = session.upload_bytes(input_view.bytes()).unwrap();
        let bank = session.allocate_zeroed_bytes(N * T::HOST_SIZE).unwrap();
        let seed = session.upload_bytes(seed_view.bytes()).unwrap();
        let completed = session.allocate_zeroed_bytes(N * T::HOST_SIZE).unwrap();
        self.kernel
            .execute_into(&[&input, &bank, &seed, &completed])
            .unwrap();
        // Both fallible reads precede either caller prefix copy, matching the host executor.
        bank.read_into_bytes(&mut self.stage).unwrap();
        completed.read_into_bytes(&mut self.output).unwrap();
        PcuHostArgument::read_write(graph::STAGE, stage)
            .bytes_mut()
            .unwrap()[..self.stage.len()]
            .copy_from_slice(&self.stage);
        PcuHostArgument::read_write(graph::OUTPUT, output)
            .bytes_mut()
            .unwrap()[..self.output.len()]
            .copy_from_slice(&self.output);
    }
}

fn geometry<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
    grid: bool,
) {
    let mut prepared = source::saved_prepare::<T, N, _>(session).unwrap();
    let mut prepared_grid = source::saved_grid_prepare::<T, N, _>(session).unwrap();
    let (mut explicit, kernel) = graph::visit(
        T::TYPE,
        u32::try_from(N).unwrap(),
        grid,
        pcu_facade::PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        |ir| {
            (
                session.prepare_host_kernel(ir).unwrap(),
                session
                    .prepare_transport_plan(MetalTransportPlan::assess(ir, T::TYPE).unwrap())
                    .unwrap(),
            )
        },
    );
    let mut native = Native {
        kernel,
        stage: vec![0; N * T::HOST_SIZE],
        output: vec![0; N * T::HOST_SIZE],
    };
    let banks = banks::<T, N>();
    for (index, bank) in banks.iter().enumerate() {
        for prior in &banks[..index] {
            assert_ne!(bytes(&bank.input), bytes(&prior.input));
        }
    }
    let mut stage = vec![T::raw(9, 83); N + 5];
    let mut output = vec![T::raw(11, 79); N + 3];
    let execute = |route: usize, bank: usize, verify: bool| {
        let bank = &banks[bank];
        match route {
            0 => {
                if grid {
                    prepared_grid(&bank.input, &bank.seed, &mut [], &mut stage, &mut output)
                        .unwrap();
                } else {
                    prepared(&bank.input, &bank.seed, &mut [], &mut stage, &mut output).unwrap();
                }
            }
            1 => {
                if grid {
                    source::saved_grid::<T, N>(
                        &bank.input,
                        &bank.seed,
                        &mut [],
                        &mut stage,
                        &mut output,
                    )
                    .unwrap();
                } else {
                    source::saved::<T, N>(
                        &bank.input,
                        &bank.seed,
                        &mut [],
                        &mut stage,
                        &mut output,
                    )
                    .unwrap();
                }
            }
            2 => explicit
                .call(&mut [
                    PcuHostArgument::read_write(graph::GHOST, &mut [] as &mut [T]),
                    PcuHostArgument::read_write(graph::OUTPUT, &mut output),
                    PcuHostArgument::read(graph::SEED, core::slice::from_ref(&bank.seed)),
                    PcuHostArgument::read_write(graph::STAGE, &mut stage),
                    PcuHostArgument::read(graph::INPUT, &bank.input),
                ])
                .unwrap(),
            3 => native.call::<T, N>(session, &bank.input, &bank.seed, &mut stage, &mut output),
            4 => {}
            _ => unreachable!("registered peer"),
        }
        if verify {
            assert_eq!(bytes(&stage), bank.stage);
            assert_eq!(bytes(&output), bank.output);
        }
        std::hint::black_box(&output);
    };
    register::<N>(criterion, T::TYPE, grid, execute);
}
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Criterion timing consumes the mutable reference in the uninstrumented build.
fn register<const N: usize>(
    criterion: &mut Criterion,
    scalar: pcu_facade::PcuScalarType,
    grid: bool,
    mut execute: impl FnMut(usize, usize, bool),
) {
    let geometry = if grid { "grid" } else { "direct" };
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    #[cfg(not(feature = "allocation-census"))]
    let mut group =
        criterion.benchmark_group(format!("metal_scalar_transport/{scalar:?}/{geometry}"));
    for (route, name) in [
        "source_prepared",
        "ordinary_pcu",
        "explicit_ir",
        "direct_native",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in 0..64 {
            execute(route, bank, true);
        }
        #[cfg(feature = "allocation-census")]
        {
            let before = SCORES.load(std::sync::atomic::Ordering::Relaxed);
            let mut total = census::Census::default();
            for bank in 0..64 {
                let ((), counts) = census::measure(|| execute(route, bank, false));
                total.alloc_calls += counts.alloc_calls;
                total.realloc_calls += counts.realloc_calls;
                total.dealloc_calls += counts.dealloc_calls;
                total.requested_bytes += counts.requested_bytes;
                execute(4, bank, true);
            }
            let scores = SCORES.load(std::sync::atomic::Ordering::Relaxed) - before;
            println!(
                "Rust allocation census/metal_scalar_transport/{:?}/{geometry}/{N}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, invocation_score_callbacks={scores} (caller Rust only; native/device heap unknown)",
                scalar,
                total.alloc_calls,
                total.realloc_calls,
                total.dealloc_calls,
                total.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        {
            let mut bank = 0;
            group.bench_function(BenchmarkId::new(name, N), |bench| {
                bench.iter(|| {
                    bank = (bank + 1) % 64;
                    execute(route, bank, false);
                });
            });
        }
        for bank in 0..64 {
            execute(route, bank, true);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
}

pub fn run(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: actual Metal required");
        return;
    }
    activity::guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let session = MetalSession::open(0).unwrap();
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
    pcu_facade::global::clear_thread_cache().unwrap();
    activity::guard();
}

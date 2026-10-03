//! Host staging, private completed output/status, escaped shape, terminal readback and Drop.
#[rustfmt::skip]
use std::{
    hint::black_box,
    rc::Rc,
};
#[rustfmt::skip]
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxSession,
    MlxEncodedArray,
    MlxCheckedProgramInput,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuTensor,
    PcuCheckedInteger,
    PcuScalar,
    PcuImplementationRequirements,
    PcuDispatchIntegerBinaryOp as Op,
    PcuRangePolicy,
    PcuHostArgument,
    PcuBindingRef,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../tensor_binary/census/census.rs"]
mod census;
#[path = "../program/program.rs"]
mod program;
use program::source;
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(device: &pcu_facade::PcuDeviceDescriptor<'_>, memory: u64) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    global::default_device_score(device, memory)
}
trait Sample: PcuCheckedInteger {
    fn small(value: u8) -> Self;
}
macro_rules! samples {
    ($($ty:ty=>$width:literal;)+) => {$(
        impl Sample for $ty {
            fn small(value:u8)->Self {
                let mut bytes = [0; $width];
                bytes[0] = value;
                Self::decode_le(bytes)
            }
        }
    )+};
}
samples! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
enum Published<T: PcuCheckedInteger> {
    Backend {
        array: MlxEncodedArray,
        shape: Rc<[usize]>,
    },
    Ordinary(PcuTensor<T>),
}
impl<T: PcuCheckedInteger> Published<T> {
    fn read(&self, output: &mut [T], count: usize) {
        match self {
            Self::Backend { array, shape } => {
                assert_eq!(shape.as_ref(), &[count]);
                array.read_into(output).unwrap();
            }
            Self::Ordinary(owner) => {
                assert_eq!(owner.shape(), &[count]);
                owner.read_into(output).unwrap();
            }
        }
    }
}
fn verify<T: Sample>(observed: &[T], wanted: &[T], sentinel: T) {
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &observed[..wanted.len()]).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), wanted).bytes()
    );
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &observed[wanted.len()..]).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &[sentinel; 2]).bytes()
    );
}
#[allow(clippy::too_many_lines)]
// Four independently prepared peers retain exact physical banks/read/drop and separate 64-call census.
#[allow(clippy::significant_drop_tightening)] // The Criterion group spans all four registrations; finish consumes it before cache cleanup.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Same Criterion context in normal, census-only context unused.
fn compare<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MlxSession,
    profile: u8,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        #[cfg(feature = "allocation-census")]
        score_device: score,
        ..Default::default()
    })
    .unwrap();
    let request = PcuImplementationRequirements::default();
    let captured = session
        .prepare_tensor_integer_program(program::capture::<T>(profile, N, request), request)
        .unwrap();
    let explicit = session
        .prepare_tensor_integer_program(program::graph::<T>(profile, N, request), request)
        .unwrap();
    let shape = captured.plan().shape_owner();
    let op = [Op::Add, Op::Sub, Op::Mul][usize::from(profile)];
    let mut native = session
        .prepare_checked_integer_control(T::TYPE, op, PcuRangePolicy::Reject, N, [N; 2], [false; 2])
        .unwrap();
    let banks: [(Vec<T>, Vec<T>, Vec<T>); 64] = std::array::from_fn(|bank| {
        let mut left = Vec::with_capacity(N);
        let mut right = Vec::with_capacity(N);
        let mut wanted = Vec::with_capacity(N);
        for lane in 0..N {
            let phase = u8::try_from((lane ^ bank) % 3 + 1).unwrap();
            let a = T::small(phase);
            let b = T::small(phase * 2);
            let expected = match profile {
                0 => a.pcu_checked_add(b),
                1 => b.pcu_checked_sub(a),
                _ => a.pcu_checked_mul(b),
            }
            .unwrap();
            left.push(a);
            right.push(b);
            wanted.push(expected);
        }
        (left, right, wanted)
    });
    let sentinel = T::small(7);
    let mut observed = vec![sentinel; N + 2];
    let mut execute = |route: usize, bank: usize, observed: &mut [T]| {
        let (left, right, _) = &banks[bank];
        if route == 3 {
            let owner = Published::Ordinary(
                match profile {
                    0 => source::add(black_box(left.as_slice()), black_box(right.as_slice())),
                    1 => source::sub(black_box(left.as_slice()), black_box(right.as_slice())),
                    _ => source::mul(black_box(left.as_slice()), black_box(right.as_slice())),
                }
                .unwrap(),
            );
            owner.read(black_box(observed), N);
            black_box(&observed);
            drop(owner);
            return;
        }
        let array = if route == 2 {
            let inputs = if profile == 1 {
                [right.as_slice(), left.as_slice()]
            } else {
                [left.as_slice(), right.as_slice()]
            };
            let (array, recovered) = native
                .execute_encoded(black_box(inputs))
                .unwrap()
                .into_parts();
            assert!(recovered.is_none());
            array
        } else {
            let prepared = if route == 0 { &captured } else { &explicit };
            let ids = prepared.plan().input_values();
            let left = PcuHostArgument::read(PcuBindingRef::new(0, 0), black_box(left));
            let right = PcuHostArgument::read(PcuBindingRef::new(0, 1), black_box(right));
            prepared
                .execute_mixed(&[
                    (
                        ids[0],
                        MlxCheckedProgramInput::Host {
                            scalar: T::TYPE,
                            bytes: left.bytes(),
                        },
                    ),
                    (
                        ids[1],
                        MlxCheckedProgramInput::Host {
                            scalar: T::TYPE,
                            bytes: right.bytes(),
                        },
                    ),
                ])
                .unwrap()
        };
        let owner = Published::<T>::Backend {
            array,
            shape: Rc::clone(&shape),
        };
        owner.read(black_box(observed), N);
        black_box(&observed);
        drop(owner);
    };
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "mlx_owned_integer_{:?}_{op:?}_host_terminal_read_drop",
        T::TYPE
    ));
    for (route, name) in [
        "captured_source",
        "explicit_graph",
        "direct_native",
        "ordinary_pcu",
    ]
    .into_iter()
    .enumerate()
    {
        for (bank, (_, _, wanted)) in banks.iter().enumerate() {
            execute(route, bank, &mut observed);
            verify(&observed, wanted, sentinel);
        }
        #[cfg(feature = "allocation-census")]
        {
            let before = SCORES.load(std::sync::atomic::Ordering::Relaxed);
            let mut total = census::Census::default();
            for (bank, (_, _, wanted)) in banks.iter().enumerate() {
                let ((), count) = census::measure(|| execute(route, bank, &mut observed));
                total.alloc_calls += count.alloc_calls;
                total.realloc_calls += count.realloc_calls;
                total.dealloc_calls += count.dealloc_calls;
                total.requested_bytes += count.requested_bytes;
                verify(&observed, wanted, sentinel);
            }
            let scored = SCORES.load(std::sync::atomic::Ordering::Relaxed) - before;
            assert_eq!(scored, 0, "warm owner call rescored candidates");
            eprintln!(
                "Rust allocation census/owned_integer/{:?}/{op:?}/{N}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, scorer_callbacks={scored} (64 changing prebuilt banks; read/drop included; native/device allocations unknown)",
                T::TYPE,
                total.alloc_calls,
                total.realloc_calls,
                total.dealloc_calls,
                total.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        group.bench_function(BenchmarkId::new(name, N), |bench| {
            let mut bank = 0;
            bench.iter(|| {
                bank = (bank + 1) % 64;
                execute(route, bank, &mut observed);
            });
        });
        for (bank, (_, _, wanted)) in banks.iter().enumerate() {
            execute(route, bank, &mut observed);
            verify(&observed, wanted, sentinel);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy::default()).unwrap();
}
fn format<T: Sample>(criterion: &mut Criterion, session: &MlxSession) {
    for profile in 0..3 {
        compare::<T, 65>(criterion, session, profile);
        compare::<T, 4096>(criterion, session, profile);
    }
}
pub fn run(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: native MLX owner benchmark needs actual Apple hardware");
        return;
    }
    activity();
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    format::<u8>(criterion, &session);
    format::<i8>(criterion, &session);
    format::<u16>(criterion, &session);
    format::<i16>(criterion, &session);
    format::<u32>(criterion, &session);
    format::<i32>(criterion, &session);
    format::<u64>(criterion, &session);
    format::<i64>(criterion, &session);
    format::<u128>(criterion, &session);
    format::<i128>(criterion, &session);
    format::<PcuU256>(criterion, &session);
    format::<PcuI256>(criterion, &session);
    format::<PcuU512>(criterion, &session);
    format::<PcuI512>(criterion, &session);
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
        println!("MLX owned-integer correctness-only observed GPU {value}% (no idle/timing claim)");
    } else {
        activity::guard();
    }
}

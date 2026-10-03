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
    PcuCheckedFloat,
    PcuImplementationRequirements,
    PcuDispatchFloatUnaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,
    PcuRangePolicy,
    PcuHostArgument,
    PcuBindingRef,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../tensor_binary/census/census.rs"]
mod census;
#[path = "../source/source.rs"]
mod source;
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(device: &pcu_facade::PcuDeviceDescriptor<'_>, memory: u64) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    global::default_device_score(device, memory)
}

trait Sample: PcuCheckedFloat {
    fn value(value: f32) -> Self;
}
macro_rules! low {($($ty:ty),+)=>{$(impl Sample for $ty {fn value(value:f32)->Self {Self::pcu_checked_from_f32(value).unwrap()}})+};}
low!(PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits);
impl Sample for f32 {
    fn value(value: f32) -> Self {
        value
    }
}
impl Sample for f64 {
    fn value(value: f32) -> Self {
        Self::from(value)
    }
}
enum Published<T: PcuCheckedFloat> {
    Backend {
        array: MlxEncodedArray,
        shape: Rc<[usize]>,
    },
    Ordinary(PcuTensor<T>),
}
impl<T: PcuCheckedFloat> Published<T> {
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
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Same Criterion context in normal, census-only context unused.
fn compare<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MlxSession,
    policy: Policy,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        float_underflow: policy,
        #[cfg(feature = "allocation-census")]
        score_device: score,
        ..Default::default()
    })
    .unwrap();
    let request = PcuImplementationRequirements {
        float_underflow: policy,
        ..PcuImplementationRequirements::default()
    };
    let capture = global::__pcu_capture_tensor_program::<T, 1, _>(
        [global::PcuSourceShape::Slice { length: N }],
        policy,
        request.numerical_mode,
        request.numerical_options,
        source::activate::__pcu_capture_entry::<T>,
    )
    .unwrap();
    let captured = session
        .prepare_checked_program(std::sync::Arc::clone(capture.program()), request)
        .unwrap();
    let mut graph = pcu_facade::dialect::tensor::Graph::try_new().unwrap();
    let input = graph.input([N], T::TYPE).unwrap();
    graph.set_numerical_options(request.numerical_options);
    let output = graph.relu(input).unwrap();
    graph
        .set_value_float_underflow_policy(output, policy)
        .unwrap();
    let explicit = session
        .prepare_checked_program(
            std::sync::Arc::new(
                graph
                    .into_selected_program(
                        &[output],
                        pcu_facade::dialect::tensor::TensorArithmeticRewritePolicy::Disabled,
                        pcu_facade::dialect::tensor::TensorArithmeticCapability::Strict,
                        pcu_facade::dialect::tensor::TensorPointwiseGroupingPolicy::Disabled,
                    )
                    .unwrap(),
            ),
            request,
        )
        .unwrap();
    let shape = captured.plan().shape_owner();
    let mut native = session
        .prepare_checked_unary_control(T::TYPE, Op::Relu, policy, PcuRangePolicy::Reject, N, false)
        .unwrap();
    let banks: [(Vec<T>, Vec<T>); 64] = std::array::from_fn(|bank| {
        let input: Vec<T> = (0..N)
            .map(|lane| {
                let phase = f32::from(u8::try_from((lane ^ bank) % 3 + 1).unwrap());
                T::value(match (lane ^ bank) % 4 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => phase,
                    _ => -phase,
                })
            })
            .collect();
        let wanted = input
            .iter()
            .map(|&value| value.pcu_checked_relu_with_policy(policy).unwrap())
            .collect();
        (input, wanted)
    });
    let sentinel = T::value(6.0);
    let mut observed = vec![sentinel; N + 2];
    let mut execute = |route: usize, bank: usize, observed: &mut [T]| {
        let (input, _) = &banks[bank];
        if route == 3 {
            let owner = Published::Ordinary(source::activate(black_box(input)).unwrap());
            owner.read(black_box(observed), N);
            black_box(&observed);
            drop(owner);
            return;
        }
        let array = if route == 2 {
            let (array, recovered) = native
                .execute_encoded(black_box(input))
                .unwrap()
                .into_parts();
            assert!(recovered.is_none());
            array
        } else {
            let prepared = if route == 0 { &captured } else { &explicit };
            let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), black_box(input));
            prepared
                .execute_mixed(&[(
                    prepared.plan().input(),
                    MlxCheckedProgramInput::Host {
                        scalar: T::TYPE,
                        bytes: bytes.bytes(),
                    },
                )])
                .unwrap()
        };
        let owner = Published::Backend {
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
        "mlx_owned_relu_{:?}_{policy:?}_host_terminal_read_drop",
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
        for (bank, (_, wanted)) in banks.iter().enumerate() {
            execute(route, bank, &mut observed);
            verify(&observed, wanted, sentinel);
        }
        #[cfg(feature = "allocation-census")]
        {
            let before = SCORES.load(std::sync::atomic::Ordering::Relaxed);
            let mut total = census::Census::default();
            for (bank, (_, wanted)) in banks.iter().enumerate() {
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
                "Rust allocation census/owned_relu/{:?}/{policy:?}/{N}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, scorer_callbacks={scored} (64 changing prebuilt banks; read/drop included; native/device allocations unknown)",
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
        for (bank, (_, wanted)) in banks.iter().enumerate() {
            execute(route, bank, &mut observed);
            verify(&observed, wanted, sentinel);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
}
fn format<T: Sample>(criterion: &mut Criterion, session: &MlxSession) {
    for policy in [
        Policy::IeeeAfterRounding,
        Policy::RejectSubnormalResult,
        Policy::AllowGradualUnderflow,
    ] {
        compare::<T, 65>(criterion, session, policy);
        compare::<T, 4096>(criterion, session, policy);
    }
}
pub fn run(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: native MLX owner benchmark needs actual Apple hardware");
        return;
    }
    activity();
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    format::<PcuF16Bits>(criterion, &session);
    format::<PcuBf16Bits>(criterion, &session);
    format::<PcuF8E4M3FnBits>(criterion, &session);
    format::<PcuF8E5M2Bits>(criterion, &session);
    format::<f32>(criterion, &session);
    format::<f64>(criterion, &session);
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
        println!("MLX owned-ReLU correctness-only observed GPU {value}% (no idle/timing claim)");
    } else {
        activity::guard();
    }
}

//! Same fresh two-input staging/private checked output/status/completion/read/drop ownership.
use super::source;
#[path = "../../support/activity/activity.rs"]
mod activity;
#[path = "sample/sample.rs"]
mod sample;
use sample::Sample;
#[cfg(feature = "allocation-census")]
#[path = "../../../../mlx/benches/checked_unary/census/census.rs"]
mod census;
#[rustfmt::skip]
use criterion::{
    Criterion,
    BenchmarkId,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuImplementationRequirements,
    PcuFloatUnderflowPolicy,
    PcuDispatchFloatBinaryOp,
    PcuMemoryPoolId,
    PcuHostArgument,
    PcuBindingRef,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{
    Graph,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalSession,
    MetalTensorBinaryPlan,
    MetalTensorInput,
};
use std::hint::black_box;
fn equal<T: Sample>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn banks<T: Sample, const N: usize>(op: PcuDispatchFloatBinaryOp) -> [(Vec<T>, Vec<T>, Vec<T>); 3] {
    [0, 1, 2].map(|phase| {
        let mut left = Vec::with_capacity(N);
        let mut right = Vec::with_capacity(N);
        let mut expected = Vec::with_capacity(N);
        for i in 0..N {
            let (a, b, c) = match op {
                PcuDispatchFloatBinaryOp::Add => [(1, 2, 3), (2, 2, 4), (3, 3, 6)][(i + phase) % 3],
                PcuDispatchFloatBinaryOp::Sub => [(3, 2, 1), (4, 1, 3), (6, 2, 4)][(i + phase) % 3],
                PcuDispatchFloatBinaryOp::Mul => [(1, 2, 2), (2, 2, 4), (2, 3, 6)][(i + phase) % 3],
                PcuDispatchFloatBinaryOp::Div => [(2, 1, 2), (4, 2, 2), (6, 2, 3)][(i + phase) % 3],
            };
            left.push(T::literal(a));
            right.push(T::literal(b));
            expected.push(T::literal(c));
        }
        (left, right, expected)
    })
}
#[allow(clippy::too_many_lines)] // Freeze four separately lowered peers then verify fresh banks around each filtered run.
fn case<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
    op: PcuDispatchFloatBinaryOp,
    underflow: PcuFloatUnderflowPolicy,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        float_underflow: underflow,
        ..Default::default()
    })
    .unwrap();
    let requirements = PcuImplementationRequirements {
        float_underflow: underflow,
        ..Default::default()
    };
    let capture = global::__pcu_capture_tensor_program::<T, 2, _>(
        [global::PcuSourceShape::Slice { length: N }; 2],
        underflow,
        requirements.numerical_mode,
        requirements.numerical_options,
        |capture, inputs| match op {
            PcuDispatchFloatBinaryOp::Add => source::add::__pcu_capture_entry::<T>(capture, inputs),
            PcuDispatchFloatBinaryOp::Sub => source::sub::__pcu_capture_entry::<T>(capture, inputs),
            PcuDispatchFloatBinaryOp::Mul => source::mul::__pcu_capture_entry::<T>(capture, inputs),
            PcuDispatchFloatBinaryOp::Div => source::div::__pcu_capture_entry::<T>(capture, inputs),
        },
    )
    .unwrap();
    let captured = session
        .prepare_tensor_binary_program(
            MetalTensorBinaryPlan::assess_program(capture.program(), requirements).unwrap(),
            PcuMemoryPoolId(143),
        )
        .unwrap();
    let mut graph = Graph::try_new().unwrap();
    let a = graph.input([N], T::TYPE).unwrap();
    let b = graph.input([N], T::TYPE).unwrap();
    let output = match op {
        PcuDispatchFloatBinaryOp::Add => graph.add(a, b),
        PcuDispatchFloatBinaryOp::Sub => graph.sub(a, b),
        PcuDispatchFloatBinaryOp::Mul => graph.mul(a, b),
        PcuDispatchFloatBinaryOp::Div => graph.div(a, b),
    }
    .unwrap();
    if T::TYPE.binary_float_format().is_some() {
        graph
            .set_value_float_underflow_policy(output, underflow)
            .unwrap();
    }
    let program = graph
        .into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let explicit = session
        .prepare_tensor_binary_program(
            MetalTensorBinaryPlan::assess_program(&program, requirements).unwrap(),
            PcuMemoryPoolId(143),
        )
        .unwrap();
    let native = session
        .prepare_native_tensor_binary_control(T::TYPE, &[N], op, underflow, PcuMemoryPoolId(143))
        .unwrap();
    let banks = banks::<T, N>(op);
    let sentinel = T::literal(6);
    let mut observed = vec![sentinel; N + 3];
    let mut execute = |route: usize, bank: usize, verify: bool| {
        if route == 0 {
            let owner = match op {
                PcuDispatchFloatBinaryOp::Add => source::add(&banks[bank].0, &banks[bank].1),
                PcuDispatchFloatBinaryOp::Sub => source::sub(&banks[bank].0, &banks[bank].1),
                PcuDispatchFloatBinaryOp::Mul => source::mul(&banks[bank].0, &banks[bank].1),
                PcuDispatchFloatBinaryOp::Div => source::div(&banks[bank].0, &banks[bank].1),
            }
            .unwrap();
            owner.read_into(&mut observed).unwrap();
            drop(owner);
        } else {
            let a = PcuHostArgument::read(PcuBindingRef::new(0, 0), &banks[bank].0);
            let b = PcuHostArgument::read(PcuBindingRef::new(0, 1), &banks[bank].1);
            let inputs = [a.bytes(), b.bytes()].map(|bytes| MetalTensorInput::HostBytes {
                scalar: T::TYPE,
                elements: N,
                bytes,
            });
            let owner = match route {
                1 => captured.execute(&inputs),
                2 => explicit.execute(&inputs),
                3 => native.execute_host([a.bytes(), b.bytes()]),
                _ => unreachable!("four physical peers"),
            }
            .unwrap();
            owner.read_into(&mut observed).unwrap();
            drop(owner);
        }
        if verify {
            equal(&observed[..N], &banks[bank].2);
            equal(&observed[N..], &[sentinel; 3]);
        }
        black_box(&observed);
    };
    let label = format!("metal_tensor_binary/{:?}/{op:?}/{underflow:?}", T::TYPE);
    let mut group = criterion.benchmark_group(&label);
    for (route, name) in [
        "ordinary_pcu_owned",
        "captured_pcu_owned",
        "explicit_graph_owned",
        "native_owned_control",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in 0..3 {
            execute(route, bank, true);
        }
        #[cfg(feature = "allocation-census")]
        census::census(&format!("{label}/{N}/{name}"), || execute(route, 1, false));
        let mut bank = 0;
        group.bench_with_input(BenchmarkId::new(name, N), &N, |bencher, _| {
            bencher.iter(|| {
                bank = (bank + 1) % 3;
                execute(route, bank, false);
            });
        });
        for bank in 0..3 {
            execute(route, bank, true);
        }
    }
    group.finish();
}
fn width<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MetalSession) {
    for op in [
        PcuDispatchFloatBinaryOp::Add,
        PcuDispatchFloatBinaryOp::Sub,
        PcuDispatchFloatBinaryOp::Mul,
        PcuDispatchFloatBinaryOp::Div,
    ] {
        let floating = T::TYPE.binary_float_format().is_some();
        if !floating && op == PcuDispatchFloatBinaryOp::Div {
            continue;
        }
        for underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            if !floating && underflow != PcuFloatUnderflowPolicy::IeeeAfterRounding {
                continue;
            }
            case::<T, N>(criterion, session, op, underflow);
        }
    }
}
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    let session = MetalSession::open(0).unwrap();
    macro_rules! widths{($($ty:ty),+)=>{$(width::<$ty,65>(criterion,&session);width::<$ty,4096>(criterion,&session);)+};}
    widths!(
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
        PcuU256,
        PcuI256,
        PcuU512,
        PcuI512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    activity::guard();
}

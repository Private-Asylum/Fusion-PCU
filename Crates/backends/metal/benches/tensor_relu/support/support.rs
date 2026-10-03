//! Equal fresh staging, checked private output/status, initialized owner, readback and Drop.
use super::source;
#[path = "../../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../../../mlx/benches/checked_unary/census/census.rs"]
mod census;
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId};
#[rustfmt::skip]
use pcu_facade::{global,PcuScalar,PcuImplementationRequirements,PcuFloatUnderflowPolicy,PcuMemoryPoolId,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuHostArgument,PcuBindingRef};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,TensorArithmeticCapability,TensorArithmeticRewritePolicy,TensorPointwiseGroupingPolicy};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalTensorPlan,MetalTensorInput};
use std::hint::black_box;
trait Sample: PcuScalar {
    const SIGN: u64;
    const ONE: u64;
    const NORMAL: u64;
    fn raw(bits: u64) -> Self;
}
macro_rules! samples{($($ty:ty,$sign:expr,$one:expr,$normal:expr;)+)=>{$(
impl Sample for $ty{const SIGN:u64=$sign;const ONE:u64=$one;const NORMAL:u64=$normal;fn raw(bits:u64)->Self{Self::from_bits(bits.try_into().unwrap())}}
)+};}
samples! {PcuF16Bits,0x8000,0x3c00,0x0400;PcuBf16Bits,0x8000,0x3f80,0x0080;PcuF8E4M3FnBits,0x80,0x38,0x08;PcuF8E5M2Bits,0x80,0x3c,0x04;f32,0x8000_0000,0x3f80_0000,0x0080_0000;}
impl Sample for f64 {
    const SIGN: u64 = 0x8000_0000_0000_0000;
    const ONE: u64 = 0x3ff0_0000_0000_0000;
    const NORMAL: u64 = 0x0010_0000_0000_0000;
    fn raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
}
fn equal<T: PcuScalar>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
#[allow(clippy::too_many_lines)] // Freeze all four matched realizations and verify each independently filtered bank.
fn width<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MetalSession) {
    let banks = [0, 1, 2].map(|phase| {
        let codes = (0..N)
            .map(|i| {
                [T::SIGN | T::ONE, T::SIGN, 0, T::ONE, T::NORMAL, T::SIGN | 1][(i + phase) % 6]
            })
            .collect::<Vec<_>>();
        let input = codes.iter().map(|&code| T::raw(code)).collect::<Vec<_>>();
        let expected = codes
            .iter()
            .map(|&code| T::raw(if code & T::SIGN != 0 { 0 } else { code }))
            .collect::<Vec<_>>();
        (input, expected)
    });
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Metal,
            float_underflow: policy,
            ..Default::default()
        })
        .unwrap();
        let requirements = PcuImplementationRequirements {
            float_underflow: policy,
            ..Default::default()
        };
        let capture = global::__pcu_capture_tensor_program::<T, 1, _>(
            [global::PcuSourceShape::Slice { length: N }],
            policy,
            requirements.numerical_mode,
            requirements.numerical_options,
            source::relu::__pcu_capture_entry::<T>,
        )
        .unwrap();
        let captured = session
            .prepare_tensor_program(
                MetalTensorPlan::assess_relu_program(capture.program(), requirements).unwrap(),
                PcuMemoryPoolId(135),
            )
            .unwrap();
        let mut graph = Graph::try_new().unwrap();
        let input = graph.input([N], T::TYPE).unwrap();
        let output = graph.relu(input).unwrap();
        graph
            .set_value_float_underflow_policy(output, policy)
            .unwrap();
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let explicit = session
            .prepare_tensor_program(
                MetalTensorPlan::assess_relu_program(&program, requirements).unwrap(),
                PcuMemoryPoolId(135),
            )
            .unwrap();
        let native = session
            .prepare_native_tensor_relu_control(T::TYPE, &[N], policy, PcuMemoryPoolId(135))
            .unwrap();
        let sentinel = T::raw(T::ONE);
        let mut observed = vec![sentinel; N + 3];
        let mut execute = |route: usize, bank: usize, verify: bool| {
            if route == 0 {
                let owner = source::relu(&banks[bank].0).unwrap();
                owner.read_into(&mut observed).unwrap();
                drop(owner);
            } else {
                let argument = PcuHostArgument::read(PcuBindingRef::new(0, 0), &banks[bank].0);
                let input = MetalTensorInput::HostBytes {
                    scalar: T::TYPE,
                    elements: N,
                    bytes: argument.bytes(),
                };
                let owner = match route {
                    1 => captured.execute(input),
                    2 => explicit.execute(input),
                    3 => native.execute_host(argument.bytes()),
                    _ => unreachable!("four registered physical peers"),
                }
                .unwrap();
                owner.read_into(&mut observed).unwrap();
                drop(owner);
            }
            if verify {
                equal(&observed[..N], &banks[bank].1);
                equal(&observed[N..], &[sentinel; 3]);
            }
            black_box(&observed);
        };
        let label = format!("metal_tensor_relu/{:?}/{policy:?}", T::TYPE);
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
}
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    let session = MetalSession::open(0).unwrap();
    macro_rules! widths{($($ty:ty),+)=>{$(width::<$ty,65>(criterion,&session);width::<$ty,4096>(criterion,&session);)+};}
    widths!(
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

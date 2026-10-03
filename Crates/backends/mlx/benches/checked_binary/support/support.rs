//! Matched fresh host inputs, private MLX output/status, completion and host prefix publication.
#[rustfmt::skip]
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
use std::hint::black_box;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxSession,
    MlxCheckedBinaryBackend,
    MlxHostKernelError,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuDispatchFloatBinaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,
    PcuRangePolicy as Range,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuHostArgument,
    PcuBindingRef,
};
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../checked_unary/census/census.rs"]
mod census;
#[path = "../../../tests/checked_binary/graph/graph.rs"]
mod graph;
#[path = "../source/source.rs"]
mod source;
trait Sample: PcuCheckedFloat + PartialEq + core::fmt::Debug {
    fn value(value: f32) -> Self;
}
macro_rules! sample {
    ($ty:ty,$convert:expr) => {
        impl Sample for $ty {
            fn value(value: f32) -> Self {
                ($convert)(value)
            }
        }
    };
}
sample!(PcuF16Bits, |value| PcuF16Bits::pcu_checked_from_f32(value)
    .unwrap());
sample!(PcuBf16Bits, |value| PcuBf16Bits::pcu_checked_from_f32(
    value
)
.unwrap());
sample!(PcuF8E4M3FnBits, |value| {
    PcuF8E4M3FnBits::pcu_checked_from_f32(value).unwrap()
});
sample!(PcuF8E5M2Bits, |value| PcuF8E5M2Bits::pcu_checked_from_f32(
    value
)
.unwrap());
sample!(f32, |value| value);
sample!(f64, f64::from);
type Source<'a, T> = Box<dyn FnMut(&[T], &[T], &mut [T]) -> Result<(), MlxHostKernelError> + 'a>;
fn prepare<T: Sample, const N: usize>(
    backend: &MlxCheckedBinaryBackend,
    op: Op,
    policy: Policy,
    range: Range,
) -> Source<'_, T> {
    match (op, policy, range) {
        (Op::Add, Policy::IeeeAfterRounding, Range::Reject) => {
            Box::new(source::add_ieee_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Add, Policy::IeeeAfterRounding, Range::Clamp) => {
            Box::new(source::add_ieee_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Reject) => {
            Box::new(source::add_tight_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Add, Policy::RejectSubnormalResult, Range::Clamp) => {
            Box::new(source::add_tight_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Reject) => {
            Box::new(source::add_gradual_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Add, Policy::AllowGradualUnderflow, Range::Clamp) => {
            Box::new(source::add_gradual_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Reject) => {
            Box::new(source::sub_ieee_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Sub, Policy::IeeeAfterRounding, Range::Clamp) => {
            Box::new(source::sub_ieee_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Reject) => {
            Box::new(source::sub_tight_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Sub, Policy::RejectSubnormalResult, Range::Clamp) => {
            Box::new(source::sub_tight_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Reject) => {
            Box::new(source::sub_gradual_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Sub, Policy::AllowGradualUnderflow, Range::Clamp) => {
            Box::new(source::sub_gradual_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Reject) => {
            Box::new(source::mul_ieee_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Mul, Policy::IeeeAfterRounding, Range::Clamp) => {
            Box::new(source::mul_ieee_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Reject) => {
            Box::new(source::mul_tight_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Mul, Policy::RejectSubnormalResult, Range::Clamp) => {
            Box::new(source::mul_tight_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Reject) => {
            Box::new(source::mul_gradual_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Mul, Policy::AllowGradualUnderflow, Range::Clamp) => {
            Box::new(source::mul_gradual_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Reject) => {
            Box::new(source::div_ieee_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Div, Policy::IeeeAfterRounding, Range::Clamp) => {
            Box::new(source::div_ieee_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Reject) => {
            Box::new(source::div_tight_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Div, Policy::RejectSubnormalResult, Range::Clamp) => {
            Box::new(source::div_tight_clamp_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Reject) => {
            Box::new(source::div_gradual_reject_prepare::<T, N, _>(backend).unwrap())
        }
        (Op::Div, Policy::AllowGradualUnderflow, Range::Clamp) => {
            Box::new(source::div_gradual_clamp_prepare::<T, N, _>(backend).unwrap())
        }
    }
}
fn expected<T: Sample>(op: Op, policy: Policy, left: &[T], right: &[T], sentinel: T) -> Vec<T> {
    let mut result: Vec<_> = left
        .iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(a, b)| {
            match op {
                Op::Add => a.pcu_checked_add_with_policy(b, policy),
                Op::Sub => a.pcu_checked_sub_with_policy(b, policy),
                Op::Mul => a.pcu_checked_mul_with_policy(b, policy),
                Op::Div => a.pcu_checked_div_with_policy(b, policy),
            }
            .unwrap()
        })
        .collect();
    result.extend([sentinel; 3]);
    result
}
#[allow(clippy::too_many_lines)] // Three matched publication peers share frozen banks, tail proof and census boundary.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Shared Criterion entry retains primary mutable harness signature.
fn operation<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
) {
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    let backend = session.checked_binary_backend();
    let mut source = prepare::<T, N>(&backend, op, policy, range);
    let mut graph = graph::fixture_profile::<T, _>(
        u32::try_from(N).unwrap(),
        op,
        policy,
        range,
        false,
        [false; 2],
        |ir| backend.prepare_host_kernel(ir),
    )
    .unwrap();
    let mut native = session
        .prepare_checked_binary_control(T::TYPE, op, policy, range, N, [N; 2], [false; 2])
        .unwrap();
    let sentinel = T::value(91.0);
    let banks = [1.0, 2.0, 3.0].map(|phase| {
        let left: Vec<_> = (0..N)
            .map(|i| T::value(phase + f32::from(u8::try_from(i % 4).unwrap())))
            .collect();
        let right: Vec<_> = (0..N).map(|i| T::value([1.0, 2.0, 4.0][i % 3])).collect();
        let expected = expected(op, policy, &left, &right, sentinel);
        (left, right, expected)
    });
    let mut output = vec![sentinel; N + 3];
    let mut execute = |route, bank: usize, verify| {
        let (left, right, expected) = &banks[bank];
        match route {
            0 => source(black_box(left), black_box(right), black_box(&mut output)).unwrap(),
            1 => graph
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(4, 1), black_box(&mut output)),
                    PcuHostArgument::read(PcuBindingRef::new(3, 2), black_box(right)),
                    PcuHostArgument::read(PcuBindingRef::new(2, 3), black_box(left)),
                ])
                .unwrap(),
            2 => native
                .call([black_box(left), black_box(right)], black_box(&mut output))
                .unwrap(),
            _ => unreachable!(),
        }
        if verify {
            assert_eq!(&output, expected);
        } else {
            black_box(output[0]);
        }
    };
    let label = format!("mlx/{:?}/{op:?}/{policy:?}/{range:?}/{N}", T::TYPE);
    for (route, name) in ["source-prepared", "explicit-graph", "direct-native"]
        .into_iter()
        .enumerate()
    {
        for bank in 0..3 {
            execute(route, bank, true);
        }
        #[cfg(not(feature = "allocation-census"))]
        {
            criterion.bench_with_input(BenchmarkId::new(&label, name), &route, |b, &route| {
                let mut bank = 0;
                b.iter(|| {
                    execute(route, bank, false);
                    bank = (bank + 1) % 3;
                });
            });
        }
        #[cfg(feature = "allocation-census")]
        {
            census::census(&format!("{label}/{name}"), || execute(route, 0, false));
        }
        for bank in 0..3 {
            execute(route, bank, true);
        }
    }
}
fn format<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MlxSession) {
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::RejectSubnormalResult,
            Policy::AllowGradualUnderflow,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                operation::<T, N>(criterion, session, op, policy, range);
            }
        }
    }
}
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Public shared Criterion interface.
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    format::<PcuF16Bits, 257>(criterion, &session);
    format::<PcuF16Bits, 65537>(criterion, &session);
    format::<PcuBf16Bits, 257>(criterion, &session);
    format::<PcuBf16Bits, 65537>(criterion, &session);
    format::<PcuF8E4M3FnBits, 257>(criterion, &session);
    format::<PcuF8E4M3FnBits, 65537>(criterion, &session);
    format::<PcuF8E5M2Bits, 257>(criterion, &session);
    format::<PcuF8E5M2Bits, 65537>(criterion, &session);
    format::<f32, 257>(criterion, &session);
    format::<f32, 65537>(criterion, &session);
    format::<f64, 257>(criterion, &session);
    format::<f64, 65537>(criterion, &session);
}

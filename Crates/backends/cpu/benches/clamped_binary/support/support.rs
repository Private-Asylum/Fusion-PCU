//! Matched transactional maps. Primitive peers prove bounded normal/overflow inputs only.
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
use super::bits::Bits;
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuDispatchFloatBinaryOp,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
};
/// Independent peer contract; no PCU arithmetic methods are called by native controls.
pub trait Native: Bits {
    fn evaluate(
        left: Self,
        right: Self,
        op: PcuDispatchFloatBinaryOp,
    ) -> Result<(Self, Option<PcuExecutionFaultKind>), PcuExecutionFaultKind>;
}
macro_rules! primitive {
    ($ty:ty) => {
        impl Native for $ty {
            fn evaluate(
                left: Self,
                right: Self,
                op: PcuDispatchFloatBinaryOp,
            ) -> Result<(Self, Option<PcuExecutionFaultKind>), PcuExecutionFaultKind> {
                if !left.is_finite() || !right.is_finite() {
                    return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
                }
                if op == PcuDispatchFloatBinaryOp::Div && right == 0.0 {
                    return Err(PcuExecutionFaultKind::DivideByZero);
                }
                let result = match op {
                    PcuDispatchFloatBinaryOp::Add => left + right,
                    PcuDispatchFloatBinaryOp::Sub => left - right,
                    PcuDispatchFloatBinaryOp::Mul => left * right,
                    PcuDispatchFloatBinaryOp::Div => left / right,
                };
                if result.is_infinite() {
                    Ok((
                        <$ty>::MAX.copysign(result),
                        Some(PcuExecutionFaultKind::ArithmeticOverflow),
                    ))
                } else {
                    Ok((result, None))
                }
            }
        }
    };
}
// This bounded workload has no tiny results. Finite scans are NOT an IEEE underflow oracle.
primitive!(f32);
primitive!(f64);
macro_rules! low {
    ($ty:ty) => {
        impl Native for $ty {
            fn evaluate(
                left: Self,
                right: Self,
                op: PcuDispatchFloatBinaryOp,
            ) -> Result<(Self, Option<PcuExecutionFaultKind>), PcuExecutionFaultKind> {
                <Self as super::oracle::Low>::FORMAT
                    .evaluate_clamped(
                        u16::try_from(Bits::bits(left)).unwrap(),
                        u16::try_from(Bits::bits(right)).unwrap(),
                        op,
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    )
                    .map(|(bits, fault)| (<Self as Bits>::from_bits(u64::from(bits)), fault))
            }
        }
    };
}
low!(PcuF16Bits);
low!(PcuBf16Bits);
low!(PcuF8E4M3FnBits);
low!(PcuF8E5M2Bits);
fn native<T: Native, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
    op: PcuDispatchFloatBinaryOp,
) -> Result<(), PcuExecutionFault> {
    assert!(left.len() >= N && right.len() >= N && output.len() >= N);
    let mut recovered = None;
    for index in 0..N {
        let (_, range) =
            T::evaluate(left[index], right[index], op).map_err(|kind| PcuExecutionFault {
                recovered: false,
                kind,
                invocation_id: u64::try_from(index).unwrap(),
            })?;
        if let Some(kind) = range {
            recovered.get_or_insert_with(|| PcuExecutionFault {
                recovered: true,
                kind,
                invocation_id: u64::try_from(index).unwrap(),
            });
        }
    }
    for (index, value) in output.iter_mut().take(N).enumerate() {
        *value = T::evaluate(left[index], right[index], op)
            .expect("whole-map preflight succeeded")
            .0;
    }
    recovered.map_or(Ok(()), Err)
}
#[allow(clippy::too_many_lines)] // Matched route proof, caller census and timed loop share one frozen workload.
pub fn compare<T: Native, const N: usize>(
    criterion: &mut Criterion,
    op: PcuDispatchFloatBinaryOp,
    mut prepared: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut ordinary: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut graph: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let max = T::from_bits(T::MAX);
    let negative_max = T::from_bits(T::MAX | T::SIGN);
    let operand = T::from_bits(match op {
        PcuDispatchFloatBinaryOp::Add => T::MAX,
        PcuDispatchFloatBinaryOp::Sub => T::MAX | T::SIGN,
        PcuDispatchFloatBinaryOp::Mul => T::ONE + T::MIN_NORMAL,
        PcuDispatchFloatBinaryOp::Div => T::ONE - T::MIN_NORMAL,
    });
    let sentinel = T::from_bits(T::ONE + 1);
    let mut left = vec![max; N];
    let right = vec![operand; N];
    let mut output = vec![sentinel; N + 3];
    let mut expected = output.clone();
    let outcome = native::<T, N>(&left, &right, &mut expected, op);
    assert!(matches!(outcome, Err(fault) if fault.recovered));
    let mut invoke = |route, left: &[T], right: &[T], output: &mut [T]| match route {
        0 => prepared(left, right, output),
        1 => ordinary(left, right, output),
        2 => graph(left, right, output),
        _ => native::<T, N>(left, right, output, op),
    };
    for initial in [max, negative_max] {
        left[0] = initial;
        let outcome = native::<T, N>(&left, &right, &mut expected, op);
        for route in 0..4 {
            assert_eq!(invoke(route, &left, &right, &mut output), outcome);
            assert_eq!(output, expected);
        }
    }
    // Fatal after a recovered lane preserves the entire caller-owned map on every route.
    let mut invalid = right.clone();
    invalid[N - 1] = T::from_bits(T::SIGN - 1);
    let before = output.clone();
    for route in 0..4 {
        assert!(
            matches!(invoke(route, &left, &invalid, &mut output), Err(fault) if !fault.recovered)
        );
        assert_eq!(output, before);
    }
    let scores = super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed);
    for route in 0..4 {
        let counts = super::ffi::count_heap(|| {
            for _ in 0..256 {
                left[0] = if left[0] == max { negative_max } else { max };
                black_box(invoke(
                    route,
                    black_box(&left),
                    black_box(&right),
                    black_box(&mut output),
                ))
                .ok();
                black_box(&output);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(
            super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed),
            scores
        );
        assert_eq!(output[N..], [sentinel; 3]);
    }
    {
        let mut group = criterion.benchmark_group(format!(
            "cpu_clamped_binary/{}/{op:?}/N{N}",
            core::any::type_name::<T>()
        ));
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        for (route, name) in [
            "source_prepared",
            "source_ordinary",
            "explicit_graph_diagnostic",
            "native_bounded_overflow_control",
        ]
        .into_iter()
        .enumerate()
        {
            group.bench_function(name, |bencher| {
                bencher.iter(|| {
                    left[0] = if left[0] == max { negative_max } else { max };
                    black_box(invoke(
                        route,
                        black_box(&left),
                        black_box(&right),
                        black_box(&mut output),
                    ))
                    .ok();
                    black_box(&output);
                });
            });
        }
        group.finish();
    }
    assert_eq!(
        super::COLD_SCORES.load(std::sync::atomic::Ordering::Relaxed),
        scores
    );
}

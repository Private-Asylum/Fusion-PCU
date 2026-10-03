//! Matched checked native publication and validation outside the measured interval.
use core::fmt::Debug;
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{Criterion, Throughput};
#[rustfmt::skip]
use fusion_pcu_cpu::PcuCpuPreparedBinaryError;
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuDispatchFloatBinaryOp,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};

fn evaluate<T: PcuCheckedFloat>(
    op: PcuDispatchFloatBinaryOp,
    lhs: T,
    rhs: T,
) -> Result<T, PcuExecutionFaultKind> {
    match op {
        PcuDispatchFloatBinaryOp::Add => lhs.pcu_checked_add(rhs),
        PcuDispatchFloatBinaryOp::Sub => lhs.pcu_checked_sub(rhs),
        PcuDispatchFloatBinaryOp::Mul => lhs.pcu_checked_mul(rhs),
        PcuDispatchFloatBinaryOp::Div => lhs.pcu_checked_div(rhs),
    }
}
fn native<T: PcuCheckedFloat, const N: usize>(
    op: PcuDispatchFloatBinaryOp,
    lhs: &[T],
    rhs: &[T],
    output: &mut [T],
) -> Result<(), PcuCpuPreparedBinaryError> {
    if lhs.len() < N || rhs.len() < N || output.len() < N {
        return Err(PcuCpuPreparedBinaryError::InvalidArguments);
    }
    for index in 0..N {
        evaluate(op, lhs[index], rhs[index]).map_err(|kind| {
            PcuCpuPreparedBinaryError::Fault(PcuExecutionFault {
                kind,
                invocation_id: index as u64,
                recovered: false,
            })
        })?;
    }
    for index in 0..N {
        output[index] = evaluate(op, lhs[index], rhs[index]).expect("successful checked preflight");
    }
    Ok(())
}

pub fn compare<T: PcuCheckedFloat + Debug + PartialEq, const N: usize>(
    criterion: &mut Criterion,
    name: &str,
    op: PcuDispatchFloatBinaryOp,
    mut source: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuCpuPreparedBinaryError>,
    mut graph: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuCpuPreparedBinaryError>,
    values: [T; 5],
) {
    let [zero, one, two, three, nan] = values;
    let mut lhs = vec![three; N];
    let rhs = vec![two; N];
    let mut output = vec![zero; N + 2];
    let mut expected = vec![zero; N + 2];
    native::<T, N>(op, &lhs, &rhs, &mut expected).unwrap();
    source(&lhs, &rhs, &mut output).unwrap();
    assert_eq!(output, expected);
    graph(&lhs, &rhs, &mut output).unwrap();
    assert_eq!(output, expected);
    lhs[N / 2] = nan;
    output.fill(three);
    let expected_fault = native::<T, N>(op, &lhs, &rhs, &mut output);
    assert!(expected_fault.is_err());
    assert_eq!(source(&lhs, &rhs, &mut output), expected_fault);
    assert_eq!(graph(&lhs, &rhs, &mut output), expected_fault);
    assert_eq!(output, vec![three; N + 2]);
    lhs[N / 2] = three;
    expected[N..].fill(three);
    source(&lhs, &rhs, &mut output).unwrap();
    assert_eq!(output, expected);
    let mut group = criterion.benchmark_group(format!("cpu_checked_binary/{name}/{op:?}/{N}"));
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("source_prepared_host", |bencher| {
        bencher.iter(|| {
            lhs[0] = if lhs[0] == one { two } else { one };
            source(black_box(&lhs), black_box(&rhs), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.bench_function("explicit_graph_diagnostic", |bencher| {
        bencher.iter(|| {
            lhs[0] = if lhs[0] == one { two } else { one };
            graph(black_box(&lhs), black_box(&rhs), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.bench_function("native_checked_host", |bencher| {
        bencher.iter(|| {
            lhs[0] = if lhs[0] == one { two } else { one };
            native::<T, N>(op, black_box(&lhs), black_box(&rhs), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.finish();
}

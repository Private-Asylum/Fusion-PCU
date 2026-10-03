//! Checked conversion native peer at the identical mixed-schema transactional host boundary.
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{Criterion,Throughput};
use fusion_pcu_cpu::PcuCpuHostError;
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuExecutionFault,PcuExecutionFaultKind};
fn native<T: PcuScalar, U: PcuScalar, const N: usize>(
    input: &[T],
    output: &mut [U],
    evaluate: fn(T) -> Result<U, PcuExecutionFaultKind>,
) -> Result<(), PcuCpuHostError> {
    assert!(input.len() >= N && output.len() >= N);
    for (index, value) in input[..N].iter().copied().enumerate() {
        evaluate(value).map_err(|kind| {
            PcuCpuHostError::Fault(PcuExecutionFault {
                kind,
                invocation_id: index as u64,
                recovered: false,
            })
        })?;
    }
    for (value, destination) in input[..N].iter().copied().zip(&mut output[..N]) {
        *destination = evaluate(value).expect("checked preflight");
    }
    Ok(())
}
pub fn compare<T: PcuScalar, U: PcuScalar, const N: usize>(
    criterion: &mut Criterion,
    mut source: impl FnMut(&[T], &mut [U]) -> Result<(), PcuCpuHostError>,
    mut graph: impl FnMut(&[T], &mut [U]) -> Result<(), PcuCpuHostError>,
    evaluate: fn(T) -> Result<U, PcuExecutionFaultKind>,
    values: [T; 3],
    sentinel: U,
) {
    let [one, two, nan] = values;
    let mut input = vec![one; N];
    let mut output = vec![sentinel; N + 2];
    let mut expected = output.clone();
    native::<T, U, N>(&input, &mut expected, evaluate).unwrap();
    source(&input, &mut output).unwrap();
    for (a, b) in output.iter().zip(&expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
    graph(&input, &mut output).unwrap();
    input[N / 2] = nan;
    let before = output.clone();
    let error = native::<T, U, N>(&input, &mut output, evaluate);
    assert!(error.is_err());
    assert_eq!(source(&input, &mut output), error);
    assert_eq!(graph(&input, &mut output), error);
    for (a, b) in output.iter().zip(&before) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
    input[N / 2] = one;
    source(&input, &mut output).unwrap();
    let mut group = criterion.benchmark_group(format!(
        "cpu_checked_conversion/{:?}_to_{:?}/{N}",
        T::TYPE,
        U::TYPE
    ));
    group.throughput(Throughput::Elements(N as u64));
    let mut next = false;
    group.bench_function("source_prepared_host", |bencher| {
        bencher.iter(|| {
            next = !next;
            input[0] = if next { one } else { two };
            source(black_box(&input), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.bench_function("explicit_graph_diagnostic", |bencher| {
        bencher.iter(|| {
            next = !next;
            input[0] = if next { one } else { two };
            graph(black_box(&input), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.bench_function("native_checked_host", |bencher| {
        bencher.iter(|| {
            next = !next;
            input[0] = if next { one } else { two };
            native::<T, U, N>(black_box(&input), black_box(&mut output), evaluate).unwrap();
            black_box(&output);
        });
    });
    group.finish();
}

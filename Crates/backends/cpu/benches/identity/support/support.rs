//! Equivalent native dense host copy with complete validation and changing inputs.
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{Criterion, Throughput};
use pcu_facade::PcuScalar;
use fusion_pcu_cpu::PcuCpuHostError;
fn native<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) -> Result<(), ()> {
    if input.len() < N || output.len() < N {
        return Err(());
    }
    output[..N].copy_from_slice(&input[..N]);
    Ok(())
}
pub fn compare<T: PcuScalar, const N: usize>(
    criterion: &mut Criterion,
    mut call: impl FnMut(&[T], &mut [T]) -> Result<(), PcuCpuHostError>,
    mut graph: impl FnMut(&[T], &mut [T]) -> Result<(), PcuCpuHostError>,
    values: [T; 2],
) {
    let mut input = vec![values[0]; N];
    let mut output = vec![values[1]; N + 2];
    call(&input, &mut output).unwrap();
    for (index, value) in output.iter().enumerate() {
        assert_eq!(
            value.encode_le().as_ref(),
            values[usize::from(index >= N)].encode_le().as_ref()
        );
    }
    graph(&input, &mut output).unwrap();
    native::<T, N>(&input, &mut output).unwrap();
    assert!(call(&input[..N - 1], &mut output).is_err());
    for value in &output[..N] {
        assert_eq!(value.encode_le().as_ref(), values[0].encode_le().as_ref());
    }
    let mut group = criterion.benchmark_group(format!("cpu_identity/{:?}/{N}", T::TYPE));
    group.throughput(Throughput::Bytes((N * T::HOST_SIZE) as u64));
    let mut next = 0;
    group.bench_function("source_prepared_host", |bencher| {
        bencher.iter(|| {
            next ^= 1;
            input[0] = values[next];
            call(black_box(&input), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.bench_function("explicit_graph_diagnostic", |bencher| {
        bencher.iter(|| {
            next ^= 1;
            input[0] = values[next];
            graph(black_box(&input), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.bench_function("native_host_copy", |bencher| {
        bencher.iter(|| {
            next ^= 1;
            input[0] = values[next];
            native::<T, N>(black_box(&input), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.finish();
}

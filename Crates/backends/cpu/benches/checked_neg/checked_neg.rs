//! Full synchronous host-boundary source/native comparisons, without GPU work.

use std::hint::black_box;
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedNeg,
    PcuCpuImplementation,
    PcuCpuProcessor,
};
use pcu_facade::pcu;

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

fn compare<const N: usize>(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("cpu_checked_neg_host");
    group.throughput(Throughput::Elements(N as u64));
    let processor = PcuCpuProcessor::detect();
    for implementation in [
        PcuCpuImplementation::Scalar,
        PcuCpuImplementation::Sse2,
        PcuCpuImplementation::Avx2,
        PcuCpuImplementation::Neon,
    ] {
        let Ok(backend) = PcuCpuCheckedNeg::new(processor, implementation) else {
            continue;
        };
        let mut call = negate_prepare::<N, _>(&backend).expect("source preparation");
        let mut input = std::vec![1.0_f32; N];
        let mut output = std::vec![0.0_f32; N];
        // Verify the complete host boundary before timing this implementation.
        call(&input, &mut output).expect("source correctness check");
        assert!(
            input
                .iter()
                .zip(&output)
                .all(|(value, result)| result.to_bits() == value.to_bits() ^ 0x8000_0000)
        );
        group.bench_with_input(
            BenchmarkId::new(format!("source_{implementation:?}"), N),
            &N,
            |bencher, _| {
                bencher.iter(|| {
                    input[0] = if input[0].to_bits() == 1.0_f32.to_bits() {
                        2.0
                    } else {
                        1.0
                    };
                    call(black_box(&input), black_box(&mut output))
                        .expect("checked source execution");
                    black_box(&output);
                });
            },
        );
    }
    let mut input = std::vec![1.0_f32; N];
    let mut output = std::vec![0.0_f32; N];
    native_checked(&input, &mut output).expect("native correctness check");
    assert!(
        input
            .iter()
            .zip(&output)
            .all(|(value, result)| result.to_bits() == value.to_bits() ^ 0x8000_0000)
    );
    // Matched native semantics: finite validation, first fault provenance and transactional
    // outputs. Both paths time changed input, checking, computation and synchronous completion.
    group.bench_with_input(BenchmarkId::new("native_checked", N), &N, |bencher, _| {
        bencher.iter(|| {
            input[0] = if input[0].to_bits() == 1.0_f32.to_bits() {
                2.0
            } else {
                1.0
            };
            let values = black_box(&input);
            native_checked(values, black_box(&mut output)).expect("checked native execution");
            black_box(&output);
        });
    });
    group.finish();
}

fn native_checked(input: &[f32], output: &mut [f32]) -> Result<(), usize> {
    if let Some(invocation) = input.iter().position(|value| !value.is_finite()) {
        return Err(invocation);
    }
    for (value, result) in input.iter().zip(output) {
        *result = f32::from_bits(value.to_bits() ^ 0x8000_0000);
    }
    Ok(())
}

fn benchmarks(criterion: &mut Criterion) {
    compare::<17>(criterion);
    compare::<4096>(criterion);
    compare::<1_048_576>(criterion);
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);

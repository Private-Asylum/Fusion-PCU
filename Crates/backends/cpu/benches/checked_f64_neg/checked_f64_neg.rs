//! Actual annotated F64 Neg, explicit prepared host, and matched checked-native full wall.

use std::hint::black_box;
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuHostBackend,
    PcuCpuImplementation,
    PcuCpuProcessor,
};
use pcu_facade::PcuHostKernelBackend;

#[path = "../../tests/prepared_f64_neg/source/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;

#[global_allocator]
static ALLOCATOR: support::RustAllocator = support::RustAllocator;

fn compare<const N: usize>(criterion: &mut Criterion) {
    let processor = PcuCpuProcessor::detect();
    let bindings = source::negate_bindings();
    let builder = source::negate_ir::<N>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut group = criterion.benchmark_group(format!("cpu_checked_f64_neg_host/{N}"));
    group.throughput(Throughput::Elements(N as u64));
    for implementation in [
        PcuCpuImplementation::Scalar,
        PcuCpuImplementation::Sse2,
        PcuCpuImplementation::Avx2,
        PcuCpuImplementation::Avx,
        PcuCpuImplementation::Avx512,
        PcuCpuImplementation::Neon,
    ] {
        let Ok(backend) = PcuCpuHostBackend::new(processor, implementation) else {
            continue;
        };
        // Actual source preparation and exact explicit preparation from that same source IR.
        // Both are cold and remain outside Criterion's warm full host-wall measurements.
        let (source_result, source_cold) =
            support::census(|| source::negate_prepare::<N, _>(&backend));
        let mut source_call = source_result.unwrap();
        let (prepared_result, prepared_cold) =
            support::census(|| backend.prepare_host_kernel(&kernel));
        let mut prepared = prepared_result.unwrap();
        support::verify::<N>(&mut source_call);
        support::verify::<N>(|input, output| support::invoke(&mut prepared, input, output));
        let mut input = std::vec![1.0; N];
        let mut output = std::vec![0.0; N + 3];
        let (source_result, source_warm) = support::census(|| source_call(&input, &mut output));
        source_result.unwrap();
        let (prepared_result, prepared_warm) =
            support::census(|| support::invoke(&mut prepared, &input, &mut output));
        prepared_result.unwrap();
        input[5] = f64::NAN;
        let (source_fault, source_fault_heap) =
            support::census(|| source_call(&input, &mut output));
        let (prepared_fault, prepared_fault_heap) =
            support::census(|| support::invoke(&mut prepared, &input, &mut output));
        assert_eq!(source_fault, prepared_fault);
        assert!(source_fault.is_err());
        for counts in [
            source_cold,
            prepared_cold,
            source_warm,
            prepared_warm,
            source_fault_heap,
            prepared_fault_heap,
        ] {
            assert_eq!(counts, support::Allocations::default());
        }
        eprintln!(
            "Rust current-thread allocation census N={N} {implementation:?}: source cold={source_cold:?}, prepared cold={prepared_cold:?}, source warm={source_warm:?}, prepared warm={prepared_warm:?}, source fault={source_fault_heap:?}, prepared fault={prepared_fault_heap:?}"
        );
        input[5] = 1.0;
        group.bench_function(format!("source_{implementation:?}"), |bencher| {
            bencher.iter(|| {
                support::change(&mut input);
                source_call(black_box(&input), black_box(&mut output)).unwrap();
                black_box(&output);
            });
        });
        group.bench_function(format!("prepared_{implementation:?}"), |bencher| {
            bencher.iter(|| {
                support::change(&mut input);
                support::invoke(&mut prepared, black_box(&input), black_box(&mut output)).unwrap();
                black_box(&output);
            });
        });
    }
    support::verify::<N>(support::native::<N>);
    let mut input = std::vec![1.0; N];
    let mut output = std::vec![0.0; N + 3];
    let (result, native_warm) = support::census(|| support::native::<N>(&input, &mut output));
    result.unwrap();
    input[5] = f64::NAN;
    let (result, native_fault) = support::census(|| support::native::<N>(&input, &mut output));
    assert!(result.is_err());
    assert_eq!(native_warm, support::Allocations::default());
    assert_eq!(native_fault, support::Allocations::default());
    eprintln!(
        "Rust current-thread allocation census N={N} native warm={native_warm:?}, native fault={native_fault:?}"
    );
    input[5] = 1.0;
    group.bench_function("native_checked", |bencher| {
        bencher.iter(|| {
            support::change(&mut input);
            support::native::<N>(black_box(&input), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.finish();
}

fn benchmarks(criterion: &mut Criterion) {
    eprintln!(
        "Timed boundary: changing input + complete schema/numeric checking + computation + synchronous completion + output observation; Criterion elapsed wall. Cold setup/oracles and current-thread Rust allocation scopes are outside timing. Rust allocator census excludes Criterion/setup containers, other threads and OS/native heap allocation APIs."
    );
    compare::<17>(criterion);
    compare::<4096>(criterion);
    compare::<1_048_576>(criterion);
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);

//! Four statically generic native caller boundaries; untimed semantics and changing-call census.
#[rustfmt::skip]
use pcu_facade::{
    PcuImplementationRequirements,
    PcuScalar,
};
use super::{bytes, ffi, COLD_SCORES};
use std::sync::atomic::Ordering;
const N: usize = 47;
pub fn compare<T: PcuScalar>(
    criterion: &mut criterion::Criterion,
    shape: &str,
    requirements: PcuImplementationRequirements,
    source: impl FnMut(&[T], &T, &mut [T], &mut [T]),
    ordinary: impl FnMut(&[T], &T, &mut [T], &mut [T]),
    graph: impl FnMut(&[T], &T, &mut [T], &mut [T]),
    native: impl FnMut(&[T], &T, &mut [T], &mut [T]),
) {
    let name = format!(
        "vulkan_ordered_transport/{:?}/{shape}/{:?}/{:?}/{:?}/{:?}/{:?}",
        T::TYPE,
        requirements.numerical_mode,
        requirements.numerical_options.compound_arithmetic,
        requirements.numerical_options.precision,
        requirements.float_underflow,
        requirements.range_policy
    );
    let mut group = criterion.benchmark_group(&name);
    group.throughput(criterion::Throughput::Elements(47));
    entry::<T>(&mut group, &name, "source_prepared", source);
    entry::<T>(&mut group, &name, "source_ordinary", ordinary);
    entry::<T>(&mut group, &name, "graph_prepared", graph);
    entry::<T>(&mut group, &name, "native_vulkan", native);
    group.finish();
}
fn entry<T: PcuScalar>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    route: &str,
    mut entry: impl FnMut(&[T], &T, &mut [T], &mut [T]),
) {
    let sentinel = bytes::sample::<T>(0xA5, 17);
    let mut stage = [sentinel; N + 3];
    let mut output = [sentinel; N + 5];
    let mut expected_stage = stage;
    let mut expected_output = output;
    let banks: [[T; N + 7]; 3] = core::array::from_fn(|bank| {
        core::array::from_fn(|lane| {
            bytes::sample::<T>(u8::try_from(bank).unwrap().wrapping_mul(0x55), lane)
        })
    });
    let seeds = core::array::from_fn::<_, 3, _>(|bank| {
        bytes::sample::<T>(u8::try_from(bank).unwrap().wrapping_mul(0xC3), 19)
    });
    for bank in 0..3 {
        entry(&banks[bank], &seeds[bank], &mut stage, &mut output);
        bytes::native::<T, N>(
            &banks[bank],
            &seeds[bank],
            &mut expected_stage,
            &mut expected_output,
        );
        bytes::equal(&stage, &expected_stage);
        bytes::equal(&output, &expected_output);
    }
    let mut phase = 0;
    let scores = COLD_SCORES.load(Ordering::Relaxed);
    if std::env::var_os("PCU_TRANSPORT_CENSUS").is_some() {
        let counts = ffi::count_heap(|| {
            for call in 0..64 {
                let bank = call % 3;
                entry(
                    std::hint::black_box(&banks[bank]),
                    std::hint::black_box(&seeds[bank]),
                    std::hint::black_box(&mut stage),
                    std::hint::black_box(&mut output),
                );
                bytes::native::<T, N>(
                    &banks[bank],
                    &seeds[bank],
                    &mut expected_stage,
                    &mut expected_output,
                );
                bytes::equal(&stage, &expected_stage);
                bytes::equal(&output, &expected_output);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(COLD_SCORES.load(Ordering::Relaxed), scores);
        println!("CENSUS {name}/{route} calls64 heap0 score0");
    } else {
        println!("SEMANTIC {name}/{route} phases3 saved_ssa1 tails1");
    }
    group.bench_function(route, |bench| {
        bench.iter(|| {
            let bank = phase % 3;
            phase += 1;
            entry(
                std::hint::black_box(&banks[bank]),
                std::hint::black_box(&seeds[bank]),
                std::hint::black_box(&mut stage),
                std::hint::black_box(&mut output),
            );
            bytes::native::<T, N>(
                &banks[bank],
                &seeds[bank],
                &mut expected_stage,
                &mut expected_output,
            );
            bytes::equal(&stage, &expected_stage);
            bytes::equal(&output, &expected_output);
        });
    });
}

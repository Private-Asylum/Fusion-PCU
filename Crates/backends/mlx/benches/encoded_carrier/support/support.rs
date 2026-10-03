//! Authentic source, independent graph and direct native fresh owner/read/drop controls.
#[rustfmt::skip]
use std::{hint::black_box,sync::Arc};
#[rustfmt::skip]
use pcu_facade::{PcuImplementationRequirements,PcuF16Bits,PcuBf16Bits,
    PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits,PcuU256,PcuI256,PcuU512,PcuI512};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,TensorArithmeticRewritePolicy,TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy};
use fusion_pcu_mlx::{MlxRuntime, MlxSession};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../checked_unary/census/census.rs"]
mod census;
#[path = "../../../tests/encoded_carrier/support/support.rs"]
mod oracle;
#[path = "../../../tests/encoded_carrier/source/source.rs"]
mod source;
use oracle::{Sample, compare};
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Same Criterion driver signature; census registers no timing samples.
fn format<T: Sample>(criterion: &mut Criterion, session: &MlxSession, count: usize) {
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    let requirements = PcuImplementationRequirements::default();
    let capture = pcu_facade::global::__pcu_capture_tensor_program(
        [pcu_facade::global::PcuSourceShape::Slice { length: count }],
        requirements.float_underflow,
        requirements.numerical_mode,
        requirements.numerical_options,
        source::retain::__pcu_capture_entry::<T>,
    )
    .unwrap();
    let prepared = session
        .prepare_checked_program(Arc::clone(capture.program()), requirements)
        .unwrap();
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input([count], T::TYPE).unwrap();
    let program = graph
        .into_selected_program(
            &[input],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let explicit = session
        .prepare_checked_program(Arc::new(program), requirements)
        .unwrap();
    let sentinel = T::sample(91);
    let banks: [Vec<T>; 3] = std::array::from_fn(|bank| {
        (0..count)
            .map(|index| {
                T::sample(
                    u8::try_from(index % 256)
                        .unwrap()
                        .wrapping_add(u8::try_from(bank * 23).unwrap()),
                )
            })
            .collect()
    });
    let mut output = vec![sentinel; count + 3];
    let mut execute = |route, bank: usize, verify| {
        if verify {
            output.fill(sentinel);
        }
        let owner = match route {
            0 => prepared.execute_host(black_box(&banks[bank])).unwrap(),
            1 => explicit.execute_host(black_box(&banks[bank])).unwrap(),
            2 => session.upload_encoded(black_box(&banks[bank])).unwrap(),
            _ => unreachable!("three registered owner peers"),
        };
        owner.read_into(black_box(&mut output)).unwrap();
        if verify {
            compare(&output[..count], &banks[bank]);
            compare(&output[count..], &[sentinel; 3]);
        }
        black_box(&owner);
        drop(owner);
    };
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!("mlx_exact_carrier_owned/{:?}", T::TYPE));
    for (route, name) in ["source-prepared", "explicit-graph", "direct-native"]
        .into_iter()
        .enumerate()
    {
        // Fresh verification banks/tails precede and follow every peer, including filtered runs.
        for bank in 0..3 {
            execute(route, bank, true);
        }
        #[cfg(feature = "allocation-census")]
        census::census(&format!("mlx-carrier/{:?}/{count}/{name}", T::TYPE), || {
            execute(route, 1, false);
        });
        #[cfg(not(feature = "allocation-census"))]
        group.bench_with_input(BenchmarkId::new(name, count), &count, |bench, _| {
            bench.iter(|| execute(route, 1, false));
        });
        for bank in 0..3 {
            execute(route, bank, true);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
}
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    for count in [257, 65_537] {
        macro_rules! all {($($ty:ty),+) => {$(format::<$ty>(criterion,&session,count);)+};}
        all!(
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
            f64,
            PcuF128Bits,
            PcuF256Bits
        );
    }
}

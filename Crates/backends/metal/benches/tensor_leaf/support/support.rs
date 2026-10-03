//! Matching fresh upload/copy/status/completion/owned read/drop, all22 physical carriers.
use super::source;
#[path = "../../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../../../mlx/benches/checked_unary/census/census.rs"]
mod census;
#[path = "../../../tests/source/carrier/sample/sample.rs"]
mod sample;
use sample::Sample;
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId};
#[rustfmt::skip]
use pcu_facade::{global,PcuImplementationRequirements,PcuMemoryPoolId,PcuHostArgument,PcuBindingRef,PcuU256,PcuI256,PcuU512,PcuI512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,TensorArithmeticCapability,TensorArithmeticRewritePolicy,TensorPointwiseGroupingPolicy};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalTensorPlan,MetalTensorInput};
use std::hint::black_box;
fn equal<T: Sample>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
#[allow(clippy::too_many_lines)] // Retain four matched realizations, then verify identical bank/read/drop boundaries.
fn width<T: Sample, const N: usize>(criterion: &mut Criterion, session: &MetalSession) {
    let requirements = PcuImplementationRequirements::default();
    let capture = global::__pcu_capture_tensor_program::<T, 1, _>(
        [global::PcuSourceShape::Slice { length: N }],
        requirements.float_underflow,
        requirements.numerical_mode,
        requirements.numerical_options,
        source::retain::__pcu_capture_entry::<T>,
    )
    .unwrap();
    let captured = session
        .prepare_tensor_program(
            MetalTensorPlan::assess_program(capture.program(), requirements).unwrap(),
            PcuMemoryPoolId(139),
        )
        .unwrap();
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input([N], T::TYPE).unwrap();
    let program = graph
        .into_selected_program(
            &[input],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let explicit = session
        .prepare_tensor_program(
            MetalTensorPlan::assess_program(&program, requirements).unwrap(),
            PcuMemoryPoolId(139),
        )
        .unwrap();
    let native = session
        .prepare_native_tensor_carrier_control(T::TYPE, &[N], PcuMemoryPoolId(139))
        .unwrap();
    let banks = [1, 199, 911].map(|seed| (0..N).map(|i| T::pattern(seed + i)).collect::<Vec<_>>());
    let sentinel = T::pattern(97);
    let mut observed = vec![sentinel; N + 3];
    let mut execute = |route: usize, bank: usize, verify: bool| {
        if route == 0 {
            let owner = source::retain(&banks[bank]).unwrap();
            owner.read_into(&mut observed).unwrap();
            drop(owner);
        } else {
            let argument = PcuHostArgument::read(PcuBindingRef::new(0, 0), &banks[bank]);
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
            equal(&observed[..N], &banks[bank]);
            equal(&observed[N..], &[sentinel; 3]);
        }
        black_box(&observed);
    };
    let label = format!("metal_tensor_leaf/{:?}", T::TYPE);
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
        #[cfg(feature = "api-census")]
        fusion_pcu_metal::reset_api_call_census();
        #[cfg(feature = "allocation-census")]
        census::census(&format!("{label}/{N}/{name}"), || execute(route, 1, false));
        #[cfg(feature = "api-census")]
        report_api(&format!("{label}/{N}/{name}"));
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
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        ..Default::default()
    })
    .unwrap();
    let session = MetalSession::open(0).unwrap();
    macro_rules! widths{($($ty:ty),+)=>{$(width::<$ty,65>(criterion,&session);width::<$ty,4096>(criterion,&session);)+};}
    widths!(
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
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    activity::guard();
}

#[cfg(feature = "api-census")]
fn report_api(label: &str) {
    let calls = fusion_pcu_metal::api_call_census();
    assert_eq!(calls.session_open_attempts, 0);
    assert_eq!(calls.shader_compile_attempts, 0);
    assert_eq!(calls.buffer_create_attempts, 3);
    assert_eq!(calls.command_buffer_attempts, 1);
    assert_eq!(calls.compute_encoder_attempts, 1);
    assert_eq!(calls.commit_attempts, 1);
    assert_eq!(calls.wait_attempts, 1);
    assert_eq!(calls.status_inspections, 1);
    assert_eq!(calls.payload_read_attempts, 1);
    eprintln!(
        "Metal owned leaf API census/{label}: {calls:?} (one warm caller-thread call; explicit protocol attempts only, excludes ARC/getters/internal SDK/driver/native allocator totals)"
    );
}

//! Literal policy identity and matched complete-host boundaries, with verification outside timing.
#[rustfmt::skip]
use std::{
    rc::Rc,
    time::Instant,
};
#[cfg(not(feature = "allocation-census"))]
use std::time::Duration;
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::Throughput;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuCompoundArithmeticPolicy,
    PcuDeviceTensor,
    PcuHostArgument,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuReproducibility,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaOwnedDispatchBackend,
    CudaOwnedTensorAssessor,
    CudaRuntime,
};
#[rustfmt::skip]
use super::{
    native::Native,
    oracle::{
        self,
        Fixture,
    },
    source,
};

pub type HostCall<'call, const N: usize> = dyn FnMut(&[f32; N], &[f32; N], &mut [f32]) + 'call;

#[cfg(not(feature = "allocation-census"))]
fn measure<const N: usize>(
    iterations: u64,
    phase: &mut u64,
    fixtures: &[Fixture<N>; 2],
    mut call: impl FnMut(&[f32; N], &[f32; N], &mut [f32]),
) -> Duration {
    let mut elapsed = Duration::ZERO;
    let mut observed = vec![0.0; N];
    for _ in 0..iterations {
        let input = &fixtures[usize::try_from(*phase & 1).unwrap()];
        *phase = phase.wrapping_add(1);
        let start = Instant::now();
        call(&input.weights, &input.gradient, &mut observed);
        elapsed += start.elapsed();
        oracle::verify(&input.expected, &observed);
    }
    elapsed
}

fn cold_and_source_witnesses() {
    for error in [
        source::checked(&[1.0], &[1.0]).unwrap_err(),
        source::strict(&[1.0], &[1.0]).unwrap_err(),
        source::portable(&[1.0], &[1.0]).unwrap_err(),
        source::tight(&[1.0], &[1.0]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"), "{error:?}");
    }
    assert!(source::preserve::<0>(&[], &[]).is_err());
    assert_eq!(1.000_000_1_f32.to_bits(), 0x3f80_0001);
    let gradient = [f32::from_bits(0x3f7f_fffe)];
    let mut observed = [0.0];
    source::witness_preserve(&[1.0], &gradient)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(observed[0].to_bits(), 0);
    source::witness_optimized(&[1.0], &gradient)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(observed[0].to_bits(), (127_u32 - 46) << 23);
    // Two actual literal bodies must never reuse a stale learning-rate graph specialization.
    for _ in 0..2 {
        source::other_rate(&[1.0], &[1.0])
            .unwrap()
            .read_into(&mut observed)
            .unwrap();
        oracle::verify(&[0.5], &observed);
        source::preserve(&[1.0], &[1.0])
            .unwrap()
            .read_into(&mut observed)
            .unwrap();
        oracle::verify(&[1.25], &observed);
    }
}

fn validate<const N: usize>(call: &mut HostCall<'_, N>, nominal: &Fixture<N>) {
    for phase in 0..3 {
        let input = oracle::fixture::<N>(phase);
        let mut observed = vec![0.0; N];
        call(&input.weights, &input.gradient, &mut observed);
        oracle::verify(&input.expected, &observed);
    }
    if N == 65 {
        let mut exceptional = nominal.weights.clone();
        exceptional[0] = f32::NAN;
        let mut observed = vec![0.0; N];
        call(&exceptional, &nominal.gradient, &mut observed);
        assert!(observed[0].is_nan());
        exceptional[0] = f32::INFINITY;
        call(&exceptional, &nominal.gradient, &mut observed);
        assert!(observed[0].is_infinite());
        call(&nominal.weights, &nominal.gradient, &mut observed);
        oracle::verify(&nominal.expected, &observed);
    }
}

fn graph_witness(
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
    precision: PcuPrecisionPolicy,
) {
    let root = CudaOwnedTensorAssessor::new(Rc::clone(backend)).unwrap();
    let assessor = root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_options(options(precision));
    let w = graph.input([1], fusion_pcu::PcuScalarType::F32).unwrap();
    let g = graph.input([1], fusion_pcu::PcuScalarType::F32).unwrap();
    let rate = f32::from_bits(0x3f80_0001);
    let result = graph.sgd_update(w, g, rate).unwrap();
    let prepared = assessor
        .prepare_owned_program(
            graph
                .into_selected_program(
                    &[result],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap(),
        )
        .unwrap();
    let pool = PcuMemoryPoolId(0);
    let weights =
        PcuDeviceTensor::new([1], backend.upload_buffer(pool, &[1.0_f32]).unwrap()).unwrap();
    let gradient = PcuDeviceTensor::new(
        [1],
        backend
            .upload_buffer(pool, &[f32::from_bits(0x3f7f_fffe)])
            .unwrap(),
    )
    .unwrap();
    let mut memory = backend.memory_provider(pool);
    let outputs = assessor
        .execute_owned_program_outputs(
            &prepared,
            &[(w, &weights), (g, &gradient)],
            pool,
            &mut memory,
        )
        .unwrap();
    let mut observed = [0.0];
    backend
        .download_buffer(pool, outputs[0].1.buffer(), &mut observed)
        .unwrap();
    let expected = if precision == PcuPrecisionPolicy::Preserve {
        0
    } else {
        (127_u32 - 46) << 23
    };
    assert_eq!(observed[0].to_bits(), expected);
    let mut native = Native::new::<1>(runtime, precision, rate).unwrap();
    native
        .host(&[1.0], &[f32::from_bits(0x3f7f_fffe)], &mut observed)
        .unwrap();
    assert_eq!(observed[0].to_bits(), expected);
}

const fn options(precision: PcuPrecisionPolicy) -> PcuNumericalOptions {
    PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision,
        reproducibility: PcuReproducibility::Unspecified,
    }
}

#[allow(clippy::too_many_lines)] // Keep three matched physical boundaries together for canonical review.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Census shares the entry interface, with no Criterion timing.
fn case<const N: usize>(
    criterion: &mut Criterion,
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
    precision: PcuPrecisionPolicy,
    paired: bool,
) {
    super::activity::guard();
    let wall = Instant::now();
    let optimized = precision == PcuPrecisionPolicy::BackendOptimized;
    let pool = PcuMemoryPoolId(0);
    let root = CudaOwnedTensorAssessor::new(Rc::clone(backend)).unwrap();
    let assessor = root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_options(options(precision));
    let w = graph.input([N], fusion_pcu::PcuScalarType::F32).unwrap();
    let g = graph.input([N], fusion_pcu::PcuScalarType::F32).unwrap();
    let updated = graph.sgd_update(w, g, oracle::RATE).unwrap();
    let prepared = assessor
        .prepare_owned_program(
            graph
                .into_selected_program(
                    &[updated],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap(),
        )
        .unwrap();
    let (escaped, expected) = {
        let input = oracle::fixture::<N>(0);
        (
            source::update(&input.weights, &input.gradient, optimized).unwrap(),
            input.expected,
        )
    };
    assert_eq!(escaped.shape(), [N]);
    fusion_pcu::global::clear_thread_cache().unwrap();
    let mut observed = vec![0.0; N];
    escaped.read_into(&mut observed).unwrap();
    oracle::verify(&expected, &observed);
    drop(escaped);
    let nominal = oracle::fixture::<N>(0);
    let mut memory = backend.memory_provider(pool);
    let mut weights =
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, &*nominal.weights).unwrap()).unwrap();
    let mut gradient = PcuDeviceTensor::new(
        [N],
        backend.upload_buffer(pool, &*nominal.gradient).unwrap(),
    )
    .unwrap();
    let mut native = Native::new::<N>(runtime, precision, oracle::RATE).unwrap();
    let mut source_call = |weights: &[f32; N], gradient: &[f32; N], observed: &mut [f32]| {
        let output = source::update(weights, gradient, optimized).unwrap();
        assert_eq!(output.shape(), [N]);
        output.read_into(observed).unwrap();
        drop(output);
    };
    let mut graph_call =
        |host_weights: &[f32; N], host_gradient: &[f32; N], observed: &mut [f32]| {
            memory
                .transfer_to(
                    weights.resource_mut(),
                    0,
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), host_weights).bytes(),
                )
                .unwrap();
            memory
                .transfer_to(
                    gradient.resource_mut(),
                    0,
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), host_gradient).bytes(),
                )
                .unwrap();
            let outputs = assessor
                .execute_owned_program_outputs(
                    &prepared,
                    &[(w, &weights), (g, &gradient)],
                    pool,
                    &mut memory,
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].1.shape(), [N]);
            backend
                .download_buffer(pool, outputs[0].1.buffer(), observed)
                .unwrap();
            drop(outputs);
        };
    let mut native_call = |weights: &[f32; N], gradient: &[f32; N], observed: &mut [f32]| {
        native.host(weights, gradient, observed).unwrap();
    };
    for call in [
        &mut source_call as &mut HostCall<'_, N>,
        &mut graph_call,
        &mut native_call,
    ] {
        validate(call, &nominal);
    }
    eprintln!(
        "native SGD F32/{N}/{precision:?}: Boundary+BackendDefined+Unspecified, ratebits={:08x}; Preserve rounded product/subtract; BackendOptimized FMA. Two retained input allocations/two changing uploads, fresh full output, per-kernel event/wait/APIstatus, full readback/output release. No numerical status buffer/BLAS guard. Integer/dyadic full-output oracle outside timing.",
        oracle::RATE.to_bits()
    );
    #[cfg(feature = "allocation-census")]
    {
        let _ = (criterion, paired);
        for (label, call) in [
            ("source", &mut source_call as &mut HostCall<'_, N>),
            ("graph", &mut graph_call),
            ("native", &mut native_call),
        ] {
            super::allocations::census(
                &format!("native_sgd/F32/{N}/{precision:?}/{label}"),
                || call(&nominal.weights, &nominal.gradient, &mut observed),
            );
            oracle::verify(&nominal.expected, &observed);
        }
        eprintln!(
            "census: one fresh native output N*4, two retained inputs; caller-thread Rust counts do not measure CUDA/driver allocations"
        );
    }
    #[cfg(not(feature = "allocation-census"))]
    {
        let fixtures = [0, 1].map(oracle::fixture::<N>);
        if paired {
            super::paired::run(
                &format!("F32/{N}/{precision:?}"),
                &fixtures,
                &mut [&mut source_call, &mut graph_call, &mut native_call],
            );
            eprintln!("paired_case_wall={:?}", wall.elapsed());
            return;
        }
        let mut phases = [0; 3];
        let mut group =
            criterion.benchmark_group(format!("cuda_native_sgd_F32_{precision:?}_full_host/{N}"));
        group.sample_size(30);
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));
        group.confidence_level(0.95);
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        group.bench_function("source", |bench| {
            bench.iter_custom(|iterations| {
                measure(iterations, &mut phases[0], &fixtures, &mut source_call)
            });
        });
        group.bench_function("explicit_graph", |bench| {
            bench.iter_custom(|iterations| {
                measure(iterations, &mut phases[1], &fixtures, &mut graph_call)
            });
        });
        group.bench_function("native_same_policy", |bench| {
            bench.iter_custom(|iterations| {
                measure(iterations, &mut phases[2], &fixtures, &mut native_call)
            });
        });
        group.finish();
    }
    eprintln!(
        "diagnostic/native_sgd/F32/{N}/{precision:?}/case_wall={:?}",
        wall.elapsed()
    );
}

pub fn run(criterion: &mut Criterion) {
    let wall = Instant::now();
    super::activity::guard();
    let (_discovery, backend, runtime) = super::selection::selected_device();
    cold_and_source_witnesses();
    let paired = std::env::var("FUSION_CUDA_NATIVE_SGD_PAIRED").is_ok_and(|value| value == "1");
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        graph_witness(&backend, &runtime, precision);
        case::<65>(criterion, &backend, &runtime, precision, paired);
        case::<1_048_576>(criterion, &backend, &runtime, precision, paired);
    }
    fusion_pcu::global::clear_thread_cache().unwrap();
    eprintln!(
        "diagnostic/whole_native_sgd/process_wall={:?}",
        wall.elapsed()
    );
}

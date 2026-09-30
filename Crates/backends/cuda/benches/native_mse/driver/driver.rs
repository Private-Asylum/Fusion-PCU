//! Frozen options and matching host/device boundaries; oracle work stays outside timing.
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

type HostCall<'call, const N: usize> = dyn FnMut(&[f32; N], &[f32; N], &mut [f32; 1]) + 'call;

#[cfg(not(feature = "allocation-census"))]
fn measure<const N: usize>(
    iterations: u64,
    phase: &mut u64,
    fixtures: &[Fixture<N>; 2],
    mut call: impl FnMut(&[f32; N], &[f32; N], &mut [f32; 1]),
) -> Duration {
    let mut elapsed = Duration::ZERO;
    let mut observed = [0.0];
    for _ in 0..iterations {
        let input = &fixtures[usize::try_from(*phase & 1).unwrap()];
        *phase = phase.wrapping_add(1);
        let start = Instant::now();
        call(&input.prediction, &input.target, &mut observed);
        elapsed += start.elapsed();
        oracle::verify(input.expected, observed[0]);
    }
    elapsed
}

fn cold_rejections() {
    for error in [
        source::checked(&[1.0], &[0.0]).unwrap_err(),
        source::strict(&[1.0], &[0.0]).unwrap_err(),
        source::portable(&[1.0], &[0.0]).unwrap_err(),
        source::tight(&[1.0], &[0.0]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"), "{error:?}");
    }
    assert!(source::preserve::<0>(&[], &[]).is_err());
}

// Untimed owner/scheduling overlap across MatMul final-event completion and synchronous MSE.
fn composition(backend: &Rc<CudaOwnedDispatchBackend>, runtime: &CudaRuntime) {
    let left = [[1.0_f32, 2.0], [3.0, 4.0]];
    let right = [[2.0_f32, 1.0], [0.0, 2.0]];
    let target = [[1.0_f32, 2.0], [3.0, 4.0]];
    // Independent integer product [[2,5],[6,11]], residual units [[1,3],[3,7]].
    let expected = f32::from(1_u16 + 9 + 9 + 49) * 0.25;
    let source_output = source::project_loss(&left, &right, &target).unwrap();
    assert!(source_output.shape().is_empty());
    let mut observed = [0.0];
    source_output.read_into(&mut observed).unwrap();
    oracle::verify(expected, observed[0]);
    drop(source_output);
    let pool = PcuMemoryPoolId(0);
    let root = CudaOwnedTensorAssessor::new(Rc::clone(backend)).unwrap();
    let assessor = root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::Preserve,
        reproducibility: PcuReproducibility::Unspecified,
    });
    let left_id = graph.input([2, 2], fusion_pcu::PcuScalarType::F32).unwrap();
    let right_id = graph.input([2, 2], fusion_pcu::PcuScalarType::F32).unwrap();
    let target_id = graph.input([2, 2], fusion_pcu::PcuScalarType::F32).unwrap();
    let projected = graph.matmul(left_id, right_id).unwrap();
    let loss = graph.mean_squared_error(projected, target_id).unwrap();
    let prepared = assessor
        .prepare_owned_program(
            graph
                .into_selected_program(
                    &[loss],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap(),
        )
        .unwrap();
    let inputs = [left, right, target].map(|values| {
        PcuDeviceTensor::new(
            [2, 2],
            backend.upload_buffer(pool, values.as_flattened()).unwrap(),
        )
        .unwrap()
    });
    let mut memory = backend.memory_provider(pool);
    let outputs = assessor
        .execute_owned_program_outputs(
            &prepared,
            &[
                (left_id, &inputs[0]),
                (right_id, &inputs[1]),
                (target_id, &inputs[2]),
            ],
            pool,
            &mut memory,
        )
        .unwrap();
    assert!(outputs[0].1.shape().is_empty());
    backend
        .download_buffer(pool, outputs[0].1.buffer(), &mut observed)
        .unwrap();
    oracle::verify(expected, observed[0]);
    // Native MSE control consumes the independently known projection; it is not a native
    // MatMul chain measurement and is excluded from all primary timing/census groups.
    let mut native = Native::new::<4>(runtime, PcuPrecisionPolicy::Preserve).unwrap();
    native
        .host(
            &[2.0, 5.0, 6.0, 11.0],
            target.as_flattened().try_into().unwrap(),
            &mut observed,
        )
        .unwrap();
    oracle::verify(expected, observed[0]);
    eprintln!(
        "preflight/MatMul->MSE source+frozen graph owner overlap accepted, scalar17; native MSE known-projection control accepted (untimed)"
    );
}

fn validate<const N: usize>(call: &mut HostCall<'_, N>, nominal: &Fixture<N>) {
    for phase in 0..3 {
        let input = oracle::fixture::<N>(phase);
        let mut observed = [0.0];
        call(&input.prediction, &input.target, &mut observed);
        oracle::verify(input.expected, observed[0]);
    }
    if N == 65 {
        let mut exceptional = nominal.prediction.clone();
        exceptional[0] = f32::NAN;
        let mut observed = [0.0];
        call(&exceptional, &nominal.target, &mut observed);
        assert!(observed[0].is_nan(), "explicit vendor NaN permission");
        exceptional[0] = f32::INFINITY;
        call(&exceptional, &nominal.target, &mut observed);
        assert!(
            observed[0].is_infinite(),
            "explicit vendor infinity permission"
        );
        call(&nominal.prediction, &nominal.target, &mut observed);
        oracle::verify(nominal.expected, observed[0]);
    }
}

#[allow(clippy::too_many_lines)] // Keep source/graph/native physical resource boundaries together for review.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Shared entry interface; census intentionally skips Criterion timing.
fn case<const N: usize>(
    criterion: &mut Criterion,
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
    precision: PcuPrecisionPolicy,
) {
    super::activity::guard();
    let wall = Instant::now();
    let optimized = precision == PcuPrecisionPolicy::BackendOptimized;
    let pool = PcuMemoryPoolId(0);
    let root = CudaOwnedTensorAssessor::new(Rc::clone(backend)).unwrap();
    let assessor = root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision,
        reproducibility: PcuReproducibility::Unspecified,
    });
    let prediction_id = graph.input([N], fusion_pcu::PcuScalarType::F32).unwrap();
    let target_id = graph.input([N], fusion_pcu::PcuScalarType::F32).unwrap();
    let loss = graph.mean_squared_error(prediction_id, target_id).unwrap();
    let prepared = assessor
        .prepare_owned_program(
            graph
                .into_selected_program(
                    &[loss],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap(),
        )
        .unwrap();
    // Drop both original host arrays before checking the escaped scalar and cache release.
    let (escaped, escaped_expected) = {
        let input = oracle::fixture::<N>(0);
        (
            source::loss(&input.prediction, &input.target, optimized).unwrap(),
            input.expected,
        )
    };
    assert!(escaped.shape().is_empty());
    fusion_pcu::global::clear_thread_cache().unwrap();
    let mut escaped_value = [0.0];
    escaped.read_into(&mut escaped_value).unwrap();
    oracle::verify(escaped_expected, escaped_value[0]);
    drop(escaped);
    let nominal = oracle::fixture::<N>(0);
    let mut memory = backend.memory_provider(pool);
    // Warm hosted source retains two staging allocations; graph/native retain exactly two too.
    let mut prediction = PcuDeviceTensor::new(
        [N],
        backend.upload_buffer(pool, &*nominal.prediction).unwrap(),
    )
    .unwrap();
    let mut target =
        PcuDeviceTensor::new([N], backend.upload_buffer(pool, &*nominal.target).unwrap()).unwrap();
    let mut native = Native::new::<N>(runtime, precision).unwrap();
    let mut source_call = |prediction: &[f32; N], target: &[f32; N], observed: &mut [f32; 1]| {
        let output = source::loss(prediction, target, optimized).unwrap();
        assert!(output.shape().is_empty());
        output.read_into(observed).unwrap();
        drop(output);
    };
    let mut graph_call =
        |host_prediction: &[f32; N], host_target: &[f32; N], observed: &mut [f32; 1]| {
            memory
                .transfer_to(
                    prediction.resource_mut(),
                    0,
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), host_prediction).bytes(),
                )
                .unwrap();
            memory
                .transfer_to(
                    target.resource_mut(),
                    0,
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), host_target).bytes(),
                )
                .unwrap();
            let outputs = assessor
                .execute_owned_program_outputs(
                    &prepared,
                    &[(prediction_id, &prediction), (target_id, &target)],
                    pool,
                    &mut memory,
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert!(outputs[0].1.shape().is_empty());
            backend
                .download_buffer(pool, outputs[0].1.buffer(), observed)
                .unwrap();
            drop(outputs);
        };
    let mut native_call = |prediction: &[f32; N], target: &[f32; N], observed: &mut [f32; 1]| {
        native.host(prediction, target, observed).unwrap();
    };
    for call in [
        &mut source_call as &mut HostCall<'_, N>,
        &mut graph_call,
        &mut native_call,
    ] {
        validate(call, &nominal);
    }
    eprintln!(
        "native MSE F32/{N}/{precision:?}: Boundary+BackendDefined+Unspecified reduction order, two retained input device allocations/two changing uploads, fresh squared scratch and scalar output, separately rounded F32 difference/square kernel event/wait, cuBLAS SASUM plus F32 reciprocal scale with device sync/APIstatus/pointer-mode transitions, scalar readback and scratch/output release inside timed call. No numerical fault status allocation; no mode retuning. Oracle checks exact small integer/dyadic domain, not general vendor bit identity."
    );
    #[cfg(feature = "allocation-census")]
    {
        let _ = criterion;
        let mut observed = [0.0];
        for (label, call) in [
            ("source", &mut source_call as &mut HostCall<'_, N>),
            ("graph", &mut graph_call),
            ("native", &mut native_call),
        ] {
            super::allocations::census(
                &format!("native_mse/F32/{N}/{precision:?}/{label}"),
                || call(&nominal.prediction, &nominal.target, &mut observed),
            );
            oracle::verify(nominal.expected, observed[0]);
        }
        eprintln!(
            "census boundary: two fresh native allocations/call (N*4 scratch,4 scalar), two retained inputs; Rust caller-thread heap counts are diagnostic and do not measure CUDA/driver allocations"
        );
    }
    #[cfg(not(feature = "allocation-census"))]
    {
        let fixtures = [0, 1].map(oracle::fixture::<N>);
        let mut phases = [0; 3];
        let mut group =
            criterion.benchmark_group(format!("cuda_native_mse_F32_{precision:?}_full_host/{N}"));
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
        "diagnostic/native_mse/F32/{N}/{precision:?}/case_wall={:?}",
        wall.elapsed()
    );
}

pub fn run(criterion: &mut Criterion) {
    let wall = Instant::now();
    super::activity::guard();
    let (_discovery, backend, runtime) = super::selection::selected_device();
    cold_rejections();
    composition(&backend, &runtime);
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        case::<65>(criterion, &backend, &runtime, precision);
        case::<1_048_576>(criterion, &backend, &runtime, precision);
    }
    fusion_pcu::global::clear_thread_cache().unwrap();
    eprintln!(
        "diagnostic/whole_native_mse/process_wall={:?}",
        wall.elapsed()
    );
}

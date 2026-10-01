//! Matched retained resident inputs and fresh outputs; complete readback/oracle outside timing.
#[rustfmt::skip]
use std::{
    mem::size_of,
    rc::Rc,
    time::{
        Duration,
        Instant,
    },
};
use criterion::Criterion;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuCompoundArithmeticPolicy,
    PcuDeviceTensor,
    PcuHostArgument,
    PcuMemoryPoolId,
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
    CublasEnvironmentSnapshot,
    CublasLtMatmulDescriptor,
    CublasLtMatmulPlan,
    CublasNumericalConfig,
    CudaCompletionBatch,
    CudaOwnedDispatchBackend,
    CudaOwnedTensorAssessor,
    CudaRuntime,
};
use super::oracle::Scalar;
#[rustfmt::skip]
use super::{
    oracle,
    source,
};

type ResidentCall<'call> = dyn FnMut(usize) -> Duration + 'call;

#[allow(clippy::too_many_lines)] // Keep three matched terminal resident boundaries and oracle gates together.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))]
fn case<T: Scalar, const R: usize, const K: usize, const C: usize>(
    criterion: &mut Criterion,
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
    precision: PcuPrecisionPolicy,
) {
    super::activity::guard();
    let wall = Instant::now();
    let optimized = precision == PcuPrecisionPolicy::BackendOptimized;
    let inputs = [0, 1].map(oracle::fill::<T, R, K, C>);
    let expected = inputs
        .each_ref()
        .map(|(left, right)| oracle::expected(left, right));
    let source_inputs = inputs.each_ref().map(|(left, right)| {
        (
            source::resident_identity::<T, R, K>(left).unwrap(),
            source::resident_identity::<T, K, C>(right).unwrap(),
        )
    });
    // Release host staging and identity executables while keeping the four resident owners.
    fusion_pcu::global::clear_thread_cache().unwrap();
    let pool = PcuMemoryPoolId(0);
    let graph_inputs = inputs.each_ref().map(|(left, right)| {
        (
            PcuDeviceTensor::new(
                [R, K],
                backend.upload_buffer(pool, left.as_flattened()).unwrap(),
            )
            .unwrap(),
            PcuDeviceTensor::new(
                [K, C],
                backend.upload_buffer(pool, right.as_flattened()).unwrap(),
            )
            .unwrap(),
        )
    });
    let native_inputs = inputs.each_ref().map(|(left, right)| {
        let mut left_buffer = runtime.allocate(R * K * size_of::<T>()).unwrap();
        let mut right_buffer = runtime.allocate(K * C * size_of::<T>()).unwrap();
        left_buffer
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), left.as_flattened()).bytes())
            .unwrap();
        right_buffer
            .copy_from(
                PcuHostArgument::read(PcuBindingRef::new(0, 1), right.as_flattened()).bytes(),
            )
            .unwrap();
        (left_buffer, right_buffer)
    });
    let root = CudaOwnedTensorAssessor::new(Rc::clone(backend)).unwrap();
    let assessor = root.assessor();
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision,
        reproducibility: PcuReproducibility::Unspecified,
    });
    let left_id = graph.input([R, K], T::TYPE).unwrap();
    let right_id = graph.input([K, C], T::TYPE).unwrap();
    let output_id = graph.matmul(left_id, right_id).unwrap();
    let program = graph
        .into_selected_program(
            &[output_id],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let cold = Instant::now();
    let prepared = assessor.prepare_owned_program(program).unwrap();
    eprintln!(
        "resident graph cold={:?} identities={:?}",
        cold.elapsed(),
        prepared.native_matmul_implementations()
    );
    let mut memory = backend.memory_provider(pool);
    let stream = runtime.create_stream().unwrap();
    let description = CublasLtMatmulDescriptor::new(
        [R, K],
        [K, C],
        [false; 2],
        CublasNumericalConfig::new(T::TYPE, precision, CublasEnvironmentSnapshot::capture())
            .unwrap(),
    )
    .unwrap();
    let cold = Instant::now();
    let control = CublasLtMatmulPlan::prepare(runtime, &stream, description).unwrap();
    eprintln!(
        "resident control cold={:?} identity={:?}",
        cold.elapsed(),
        control.identity()
    );
    let mut source_observed = vec![T::default(); R * C];
    let mut graph_observed = vec![T::default(); R * C];
    let mut control_observed = vec![T::default(); R * C];
    let mut source_call = |bank: usize| {
        let start = Instant::now();
        let output = if optimized {
            source::optimized::<T, R, K, C>(&source_inputs[bank].0, &source_inputs[bank].1)
        } else {
            source::preserve::<T, R, K, C>(&source_inputs[bank].0, &source_inputs[bank].1)
        }
        .unwrap();
        let mut elapsed = start.elapsed();
        output.read_into(&mut source_observed).unwrap();
        oracle::verify(&expected[bank], &source_observed);
        let release = Instant::now();
        drop(output);
        elapsed += release.elapsed();
        elapsed
    };
    let mut graph_call = |bank: usize| {
        let start = Instant::now();
        let outputs = assessor
            .execute_owned_program_outputs(
                &prepared,
                &[
                    (left_id, &graph_inputs[bank].0),
                    (right_id, &graph_inputs[bank].1),
                ],
                pool,
                &mut memory,
            )
            .unwrap();
        let mut elapsed = start.elapsed();
        backend
            .download_buffer(pool, outputs[0].1.buffer(), &mut graph_observed)
            .unwrap();
        oracle::verify(&expected[bank], &graph_observed);
        let release = Instant::now();
        drop(outputs);
        elapsed += release.elapsed();
        elapsed
    };
    let mut phase_total = [Duration::ZERO; 2];
    let mut phase_count = 0_u32;
    let mut control_call = |bank: usize| {
        let start = Instant::now();
        let output = runtime.allocate(R * C * size_of::<T>()).unwrap();
        let mut batch = CudaCompletionBatch::new(&stream);
        control
            .submit_into_batch(
                &mut batch,
                &native_inputs[bank].0,
                &native_inputs[bank].1,
                &output,
            )
            .unwrap();
        let mut completion = batch.finish().unwrap();
        let submission = start.elapsed();
        let completion_start = Instant::now();
        completion.wait().unwrap();
        drop(completion);
        drop(batch);
        let completed = completion_start.elapsed();
        phase_total[0] += submission;
        phase_total[1] += completed;
        phase_count = phase_count.checked_add(1).unwrap();
        output
            .copy_to(
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut control_observed)
                    .bytes_mut()
                    .unwrap(),
            )
            .unwrap();
        oracle::verify(&expected[bank], &control_observed);
        let release = Instant::now();
        drop(output);
        submission + completed + release.elapsed()
    };
    let source_cold = Instant::now();
    source_call(0);
    eprintln!(
        "resident source cold first-call wall={:?}; includes preparation and one completed MatMul",
        source_cold.elapsed()
    );
    for bank in [0, 1, 0] {
        source_call(bank);
        graph_call(bank);
        control_call(bank);
    }
    eprintln!(
        "resident {}/{R}x{K}x{C}/{precision:?}: two changing retained input banks; fresh output, Lt submission, one final event/wait and output release timed; full readback/oracle excluded from timing and included in wall. Every selected plan retains its own cold workspace. Source and graph wrapper owners remain distinct from native control.",
        T::LABEL
    );
    #[cfg(feature = "allocation-census")]
    {
        let _ = criterion;
        for (label, call) in [
            ("source", &mut source_call as &mut ResidentCall<'_>),
            ("graph", &mut graph_call),
            ("native_lt", &mut control_call),
        ] {
            fusion_pcu_cuda::reset_cublaslt_api_census();
            super::allocations::census(
                &format!(
                    "native_matmul_resident/{}/{R}x{K}x{C}/{precision:?}/{label}",
                    T::LABEL
                ),
                || {
                    call(1);
                },
            );
            let api = fusion_pcu_cuda::cublaslt_api_census();
            eprintln!(
                "resident warm API census {label}: {api:?}; Rust census includes untimed oracle/readback callback work"
            );
            assert_eq!(api.loads, 0);
            assert_eq!(api.heuristics, 0);
            assert_eq!(api.matmuls, 1);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    {
        let mut group = criterion.benchmark_group(format!(
            "cuda_native_matmul_{}_{precision:?}_resident/{R}x{K}x{C}",
            T::LABEL
        ));
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.confidence_level(0.95);
        for (label, call) in [
            ("source", &mut source_call as &mut ResidentCall<'_>),
            ("explicit_graph", &mut graph_call),
            ("native_lt_same_policy", &mut control_call),
        ] {
            let mut bank = 0;
            group.bench_function(label, |bench| {
                bench.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iterations {
                        total += call(bank);
                        bank ^= 1;
                    }
                    total
                });
            });
        }
        group.finish();
    }
    eprintln!(
        "resident independent Lt control phases/{}/{R}x{K}x{C}/{precision:?}: calls={phase_count}, submission_mean={:?}, completion_mean={:?}, wall={:?}; diagnostic host phases, not isolated GPU throughput",
        T::LABEL,
        phase_total[0] / phase_count,
        phase_total[1] / phase_count,
        wall.elapsed()
    );
}
pub fn run(
    criterion: &mut Criterion,
    backend: &Rc<CudaOwnedDispatchBackend>,
    runtime: &CudaRuntime,
) {
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        case::<f32, 4, 2, 4>(criterion, backend, runtime, precision);
        case::<f32, 64, 96, 32>(criterion, backend, runtime, precision);
        case::<f64, 4, 2, 4>(criterion, backend, runtime, precision);
        case::<f64, 64, 96, 32>(criterion, backend, runtime, precision);
    }
}

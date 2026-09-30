//! Nominal exact-dyadic oracle with full-host source/graph/native allocation and status parity.
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
#[cfg(not(feature = "allocation-census"))]
use super::oracle::Scalar;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuBindingRef,
    PcuHostArgument,
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
    heavy::HeavyScalar,
    oracle,
    source,
};

type HostCall<'call, T, const R: usize, const K: usize, const C: usize> =
    dyn FnMut(&[[T; K]; R], &[[T; C]; K], &mut [T]) + 'call;

struct Fixture<T, const R: usize, const K: usize, const C: usize> {
    left: Box<[[T; K]; R]>,
    right: Box<[[T; C]; K]>,
    expected: Vec<T>,
}

fn fixture<T: HeavyScalar, const R: usize, const K: usize, const C: usize>(
    phase: u64,
) -> Fixture<T, R, K, C> {
    let (left, right, expected) = if K > 256 {
        let (left, right) = super::heavy::fill::<T, R, K, C>(phase);
        let expected = super::heavy::expected::<T, R, K, C>(phase);
        (left, right, expected)
    } else {
        let (left, right) = oracle::fill::<T, R, K, C>(phase);
        let expected = oracle::expected(&left, &right);
        (left, right, expected)
    };
    Fixture {
        left,
        right,
        expected,
    }
}

#[cfg(not(feature = "allocation-census"))]
fn measure<T: Scalar, const R: usize, const K: usize, const C: usize>(
    iterations: u64,
    phase: &mut u64,
    fixtures: &[Fixture<T, R, K, C>; 2],
    mut call: impl FnMut(&[[T; K]; R], &[[T; C]; K], &mut [T]),
) -> Duration {
    let mut total = Duration::ZERO;
    let mut observed = vec![T::default(); R * C];
    for _ in 0..iterations {
        let fixture = &fixtures[usize::try_from(*phase & 1).unwrap()];
        *phase = phase.wrapping_add(1);
        let start = Instant::now();
        call(&fixture.left, &fixture.right, &mut observed);
        total += start.elapsed();
        oracle::verify(&fixture.expected, &observed);
    }
    total
}

#[allow(clippy::too_many_lines)] // Keep all three full-host work boundaries visible for benchmark review.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Same Criterion entry interface; this separate build records allocations.
fn case<T: HeavyScalar, const R: usize, const K: usize, const C: usize>(
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
    let left_id = graph.input([R, K], T::TYPE).unwrap();
    let right_id = graph.input([K, C], T::TYPE).unwrap();
    let output = graph.matmul(left_id, right_id).unwrap();
    let program = graph
        .into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let prepared = assessor.prepare_owned_program(program).unwrap();
    let mut memory = backend.memory_provider(pool);
    let mut native = Native::new::<T, R, K, C>(runtime, precision).unwrap();
    let nominal = fixture::<T, R, K, C>(0);
    let (left, right) = (&nominal.left, &nominal.right);
    assert!(source::checked_boundary::<T, R, K, C>(left, right).is_err());
    assert!(source::strict_native::<T, R, K, C>(left, right).is_err());
    assert!(source::portable_native::<T, R, K, C>(left, right).is_err());
    assert!(source::tight_native::<T, R, K, C>(left, right).is_err());
    // Hosted source staging retains these two allocations after its first call. Match that
    // physical lifetime rather than compare it against repeated fresh input allocation.
    let mut graph_left = PcuDeviceTensor::new(
        [R, K],
        backend.upload_buffer(pool, left.as_flattened()).unwrap(),
    )
    .unwrap();
    let mut graph_right = PcuDeviceTensor::new(
        [K, C],
        backend.upload_buffer(pool, right.as_flattened()).unwrap(),
    )
    .unwrap();
    let mut source_call = |left: &[[T; K]; R], right: &[[T; C]; K], observed: &mut [T]| {
        let output = source::product(left, right, optimized).unwrap();
        output.read_into(observed).unwrap();
        drop(output);
    };
    let mut graph_call = |left: &[[T; K]; R], right: &[[T; C]; K], observed: &mut [T]| {
        memory
            .transfer_to(
                graph_left.resource_mut(),
                0,
                PcuHostArgument::read(PcuBindingRef::new(0, 0), left.as_flattened()).bytes(),
            )
            .unwrap();
        memory
            .transfer_to(
                graph_right.resource_mut(),
                0,
                PcuHostArgument::read(PcuBindingRef::new(0, 1), right.as_flattened()).bytes(),
            )
            .unwrap();
        let output = assessor
            .execute_owned_program_outputs(
                &prepared,
                &[(left_id, &graph_left), (right_id, &graph_right)],
                pool,
                &mut memory,
            )
            .unwrap();
        backend
            .download_buffer(pool, output[0].1.buffer(), observed)
            .unwrap();
        drop(output);
    };
    let mut native_call = |left: &[[T; K]; R], right: &[[T; C]; K], observed: &mut [T]| {
        native.host(left, right, observed).unwrap();
    };
    for phase in 0..3 {
        let input = fixture::<T, R, K, C>(phase);
        for call in [
            &mut source_call as &mut HostCall<'_, T, R, K, C>,
            &mut graph_call,
            &mut native_call,
        ] {
            let mut observed = vec![T::default(); R * C];
            call(&input.left, &input.right, &mut observed);
            oracle::verify(&input.expected, &observed);
        }
    }
    eprintln!(
        "native MatMul {}/{R}x{K}x{C}/{precision:?}: Boundary+BackendDefined, no PortableV1/strict/tight claim; matched2retained input allocations,1fresh output allocation,2uploads,one final stream event/wait/APIstatus,1readback and output release. Two input banks alternate across measured calls. No numerical fault/status buffer under this explicitly permitted vendor contract. Controlled exact dyadic or split-depth rank-two integer domain does not certify all vendor numerical behavior.",
        T::LABEL
    );
    #[cfg(feature = "allocation-census")]
    {
        let _ = criterion;
        let mut observed = vec![T::default(); R * C];
        for (label, call) in [
            ("source", &mut source_call as &mut HostCall<'_, T, R, K, C>),
            ("graph", &mut graph_call),
            ("native", &mut native_call),
        ] {
            super::allocations::census(
                &format!(
                    "native_matmul/{}/{R}x{K}x{C}/{precision:?}/{label}",
                    T::LABEL
                ),
                || call(left, right, &mut observed),
            );
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    {
        let fixtures = [0, 1].map(fixture::<T, R, K, C>);
        let mut phases = [0; 3];
        let mut group = criterion.benchmark_group(format!(
            "cuda_native_matmul_{}_{precision:?}_full_host/{R}x{K}x{C}",
            T::LABEL
        ));
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.confidence_level(0.95);
        group.throughput(Throughput::Elements(u64::try_from(R * C).unwrap()));
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
        "diagnostic/native_matmul/{}/{R}x{K}x{C}/{precision:?}/process_wall={:?}",
        T::LABEL,
        wall.elapsed()
    );
}

pub fn run(criterion: &mut Criterion) {
    let wall = Instant::now();
    super::activity::guard();
    let (_discovery, backend, runtime) = super::selection::selected_device();
    super::heavy::preflight::<f32>();
    super::heavy::preflight::<f64>();
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        case::<f32, 4, 2, 4>(criterion, &backend, &runtime, precision);
        case::<f32, 128, 256, 128>(criterion, &backend, &runtime, precision);
        case::<f32, 1024, 2048, 1024>(criterion, &backend, &runtime, precision);
        case::<f64, 4, 2, 4>(criterion, &backend, &runtime, precision);
        case::<f64, 128, 256, 128>(criterion, &backend, &runtime, precision);
        case::<f64, 1024, 2048, 1024>(criterion, &backend, &runtime, precision);
    }
    fusion_pcu::global::clear_thread_cache().unwrap();
    eprintln!(
        "diagnostic/whole_native_matmul/process_wall={:?}",
        wall.elapsed()
    );
}

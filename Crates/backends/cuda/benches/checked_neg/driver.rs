//! Matched full-host submission, terminal status and output-readback boundaries.
use std::time::Instant;
#[cfg(not(feature = "allocation-census"))]
use std::time::Duration;
#[rustfmt::skip]
use criterion::{
    Criterion,
};
#[cfg(not(feature = "allocation-census"))]
use criterion::Throughput;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuExecutionError,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaHostKernelError,
    CudaPreparedHostKernel,
};

#[rustfmt::skip]
use super::{
    graph,
    native::Native,
    oracle::{
        self,
        Scalar,
    },
};

fn call_graph<T: Scalar>(
    prepared: &mut CudaPreparedHostKernel,
    input: &[T],
    output: &mut [T],
) -> Result<(), CudaHostKernelError> {
    prepared.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
    ])
}

const fn fault(kind: PcuExecutionFaultKind) -> PcuExecutionFault {
    PcuExecutionFault {
        kind,
        invocation_id: 1,
        recovered: false,
    }
}

fn unchanged<T: Scalar>(output: &[T], original: T) {
    assert!(
        output
            .iter()
            .all(|value| value.encoding() == original.encoding())
    );
}

fn preflight<T: Scalar, const N: usize>(
    prepared: &mut CudaPreparedHostKernel,
    native: &mut Native,
) {
    let mut input = oracle::input::<T>(N, 0);
    let original = T::sample(7, 17);
    let mut output = vec![original; N];
    input[1] = T::invalid();
    let expected = fault(PcuExecutionFaultKind::InvalidFloatingOperand);
    assert!(
        matches!(T::source::<N>(&input, &mut output), Err(PcuExecutionError::ArithmeticFault(actual)) if actual == expected)
    );
    unchanged(&output, original);
    assert!(
        matches!(call_graph(prepared, &input, &mut output), Err(CudaHostKernelError::CheckedExecutionFault(actual)) if actual == expected)
    );
    unchanged(&output, original);
    native.upload(&oracle::bytes(&input));
    assert_eq!(native.submit(), 13);

    // A normal success follows the same route's fault, proving its status reset independently.
    for phase in [0, 1, 17] {
        let input = oracle::input::<T>(N, phase);
        T::source::<N>(&input, &mut output).unwrap();
        oracle::verify(&input, &output);
        call_graph(prepared, &input, &mut output).unwrap();
        oracle::verify(&input, &output);
        native.host(&oracle::bytes(&input));
        oracle::verify_bytes(&input, native.completed_bytes());
    }
    let mut input = oracle::input::<T>(N, 1);
    input[1] = T::tiny();
    T::source::<N>(&input, &mut output).unwrap();
    oracle::verify(&input, &output);
    call_graph(prepared, &input, &mut output).unwrap();
    oracle::verify(&input, &output);
    native.host(&oracle::bytes(&input));
    oracle::verify_bytes(&input, native.completed_bytes());
    output.fill(original);
    assert!(
        matches!(T::reject_subnormal::<N>(&input, &mut output), Err(PcuExecutionError::ArithmeticFault(actual)) if actual == fault(PcuExecutionFaultKind::ArithmeticUnderflow))
    );
    unchanged(&output, original);
    // Returning to default after an explicit policy call must retain its own specialization.
    T::source::<N>(&input, &mut output).unwrap();
    oracle::verify(&input, &output);
}

#[cfg(not(feature = "allocation-census"))]
fn measure<T: Scalar>(
    iterations: u64,
    count: usize,
    mut call: impl FnMut(&[T], &mut [T]),
) -> Duration {
    let mut input = oracle::input::<T>(count, 1);
    let mut output = vec![T::sample(0, 0); count];
    let mut elapsed = Duration::ZERO;
    for iteration in 0..iterations {
        input[0] = T::sample(0, iteration.wrapping_add(1));
        let start = Instant::now();
        call(&input, &mut output);
        elapsed += start.elapsed();
        oracle::verify(&input, &output);
    }
    elapsed
}

#[cfg(not(feature = "allocation-census"))]
fn measure_native<T: Scalar>(iterations: u64, count: usize, native: &mut Native) -> Duration {
    let mut input = oracle::input::<T>(count, 1);
    let mut bytes = oracle::bytes(&input);
    let mut elapsed = Duration::ZERO;
    for iteration in 0..iterations {
        input[0] = T::sample(0, iteration.wrapping_add(1));
        input[0].write(&mut bytes[..size_of::<T>()]);
        let start = Instant::now();
        native.host(&bytes);
        elapsed += start.elapsed();
        oracle::verify_bytes(&input, native.completed_bytes());
    }
    elapsed
}

#[cfg(not(feature = "allocation-census"))]
fn paired<T: Scalar, const N: usize>(prepared: &mut CudaPreparedHostKernel, native: &mut Native) {
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut source_ratios = [0.0; 36];
    let mut graph_ratios = [0.0; 36];
    for triple in 0..36 {
        let input = oracle::input::<T>(N, u64::try_from(triple).unwrap().wrapping_add(1));
        let bytes = oracle::bytes(&input);
        let mut output = vec![T::sample(0, 0); N];
        let mut times = [Duration::ZERO; 3];
        for route in orders[triple % orders.len()] {
            let start = Instant::now();
            match route {
                0 => T::source::<N>(&input, &mut output).unwrap(),
                1 => call_graph(prepared, &input, &mut output).unwrap(),
                2 => native.host(&bytes),
                _ => unreachable!(),
            }
            times[route] = start.elapsed();
            if route == 2 {
                oracle::verify_bytes(&input, native.completed_bytes());
            } else {
                oracle::verify(&input, &output);
            }
        }
        source_ratios[triple] = times[0].as_secs_f64() / times[2].as_secs_f64();
        graph_ratios[triple] = times[1].as_secs_f64() / times[2].as_secs_f64();
    }
    source_ratios.sort_by(f64::total_cmp);
    graph_ratios.sort_by(f64::total_cmp);
    eprintln!(
        "paired/checked_neg/{}/{N}: median_actual_source/native={}, graph/native={} (36 balanced triples; no ratio CI)",
        T::NAME,
        f64::midpoint(source_ratios[17], source_ratios[18]),
        f64::midpoint(graph_ratios[17], graph_ratios[18])
    );
}

pub fn profile<T: Scalar, const N: usize>(criterion: &mut Criterion) {
    super::activity::activity_guard();
    let wall = Instant::now();
    let (_discovery, backend) = super::discovery::selected_device();
    let mut prepared = graph::with_ir::<T, N, _>(|ir| backend.prepare_host_kernel(ir).unwrap());
    let mut native = T::native::<N>(&backend);
    preflight::<T, N>(&mut prepared, &mut native);
    #[cfg(feature = "allocation-census")]
    {
        let input = oracle::input::<T>(N, 1);
        let bytes = oracle::bytes(&input);
        let mut output = vec![T::sample(0, 0); N];
        super::allocations::census(&format!("neg/{}/{N}/source", T::NAME), || {
            T::source::<N>(&input, &mut output).unwrap();
        });
        oracle::verify(&input, &output);
        super::allocations::census(&format!("neg/{}/{N}/graph", T::NAME), || {
            call_graph(&mut prepared, &input, &mut output).unwrap();
        });
        oracle::verify(&input, &output);
        super::allocations::census(&format!("neg/{}/{N}/native", T::NAME), || {
            native.host(&bytes);
        });
        oracle::verify_bytes(&input, native.completed_bytes());
        std::hint::black_box(criterion);
    }
    #[cfg(not(feature = "allocation-census"))]
    {
        let mut group =
            criterion.benchmark_group(format!("cuda_checked_neg_{}_full_host", T::NAME));
        group.sample_size(20);
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.bench_function(format!("source/{N}"), |bench| {
            bench.iter_custom(|iterations| {
                measure::<T>(iterations, N, |input, output| {
                    T::source::<N>(input, output).unwrap();
                })
            });
        });
        group.bench_function(format!("explicit_graph/{N}"), |bench| {
            bench.iter_custom(|iterations| {
                measure::<T>(iterations, N, |input, output| {
                    call_graph(&mut prepared, input, output).unwrap();
                })
            });
        });
        group.bench_function(format!("native_same_checker/{N}"), |bench| {
            bench.iter_custom(|iterations| measure_native::<T>(iterations, N, &mut native));
        });
        group.finish();
    }
    #[cfg(not(feature = "allocation-census"))]
    paired::<T, N>(&mut prepared, &mut native);
    eprintln!(
        "diagnostic/checked_neg/{}/{N}/process_wall={:?}",
        T::NAME,
        wall.elapsed()
    );
}

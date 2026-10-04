//! Matched retained genuine source, explicitly authored IR and independent raw SDK arithmetic.
#[rustfmt::skip]
use std::{time::{Duration,Instant},sync::atomic::{AtomicUsize,Ordering}};
#[rustfmt::skip]
use fusion_pcu::{global,PcuBindingRef,PcuDispatchKernelIr,PcuDispatchOp,PcuHostArgument,
 PcuHostKernelBackend,PcuPreparedHostKernel,PcuExecutionError,PcuExecutionFault,
 PcuImplementationRequirements,PcuNumericalMode,PcuNumericalOptions,
 PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuReproducibility,PcuFloatUnderflowPolicy,
 PcuRangePolicy,PcuOwnedDispatchBackend,PcuDispatchSubmission,PcuInvocationShape};
use fusion_pcu_cuda::CudaOwnedDispatchBackend;
use criterion::{Criterion, Throughput};
use super::{
    oracle::{self, Format},
    source, ir,
    native::Native,
};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
pub trait FaultError: std::fmt::Debug {
    fn arithmetic(&self) -> Option<PcuExecutionFault>;
}
impl FaultError for fusion_pcu_cuda::CudaHostKernelError {
    fn arithmetic(&self) -> Option<PcuExecutionFault> {
        match self {
            Self::CheckedExecutionFault(f) => Some(*f),
            _ => None,
        }
    }
}
impl FaultError for fusion_pcu::cpu::PcuCpuHostError {
    fn arithmetic(&self) -> Option<PcuExecutionFault> {
        (*self).fault()
    }
}
type Status = Result<Option<(u32, u32, bool)>, ()>;
fn observed<E>(
    result: Result<(), E>,
    fault: impl FnOnce(&E) -> Option<PcuExecutionFault>,
) -> Status {
    match result {
        Ok(()) => Ok(None),
        Err(e) => fault(&e).map_or(Err(()), |f| {
            Ok(Some((
                u32::try_from(f.invocation_id).unwrap(),
                oracle::code(f.kind),
                f.recovered,
            )))
        }),
    }
}
#[allow(clippy::too_many_arguments)] // Each route observes the same explicit borrowed host transaction.
fn call<T: Format, P: PcuPreparedHostKernel>(
    route: &str,
    prepared: &mut P,
    native: &mut Option<Native>,
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
    dead: bool,
    range: PcuRangePolicy,
    host: &mut impl FnMut(&[T], &T, &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
) -> Status
where
    P::Error: FaultError,
{
    match route {
        "actual_source" => observed(
            host(input, seed, stage, output),
            PcuExecutionError::arithmetic_fault,
        ),
        "explicit_ir" => {
            let result = if dead {
                prepared.call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                    PcuHostArgument::read_scalar(PcuBindingRef::new(0, 1), seed),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
                ])
            } else {
                prepared.call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                    PcuHostArgument::read_scalar(PcuBindingRef::new(0, 1), seed),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), stage),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 3), output),
                ])
            };
            observed(result, FaultError::arithmetic)
        }
        "independent_sdk" => native
            .as_mut()
            .unwrap()
            .call(input, seed, (!dead).then_some(stage), output)
            .map(|f| f.map(|(index, code)| (index, code, range == PcuRangePolicy::Clamp))),
        _ => unreachable!(),
    }
}
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::significant_drop_tightening
)] // Full cold profile, transaction proof, timing boundary and census stay together.
fn case<T: Format, const N: usize, B: PcuHostKernelBackend>(
    criterion: &mut Criterion,
    backend: &B,
    cuda: Option<&CudaOwnedDispatchBackend>,
    source_ir: &PcuDispatchKernelIr<'_>,
    explicit_ir: &PcuDispatchKernelIr<'_>,
    grid: bool,
    dead: bool,
    semantics: bool,
    mut host: impl FnMut(&[T], &T, &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
) where
    B::Error: std::fmt::Debug,
    <B::Prepared as PcuPreparedHostKernel>::Error: FaultError,
{
    let request = source_ir.numerical_requirements;
    assert_eq!(request, explicit_ir.numerical_requirements);
    let source_schema =
        fusion_pcu::describe_portable_v1_checked_integer_composed_map::<4>(source_ir).unwrap();
    let explicit_schema =
        fusion_pcu::describe_portable_v1_checked_integer_composed_map::<4>(explicit_ir).unwrap();
    assert_eq!(source_schema.resources(), explicit_schema.resources());
    assert_eq!(source_schema.logical_extent, u32::try_from(N).unwrap());
    let mut prepared = backend.prepare_host_kernel(explicit_ir).unwrap();
    let mut native = cuda.map(|backend| {
        let dispatch = backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: explicit_ir,
                shape: PcuInvocationShape::invocations(
                    std::num::NonZeroU32::new(explicit_ir.entry.logical_shape[0]).unwrap(),
                ),
            })
            .unwrap();
        Native::new::<T, N>(
            backend.device_identity().device_id(),
            grid,
            request.range_policy,
            dead,
            dispatch.launch_geometry(),
        )
    });
    #[cfg(feature = "allocation-census")]
    let direct_counter = native.as_ref().map(Native::counter);
    let sentinel = oracle::small::<T>(17);
    let mut inputs = vec![oracle::small::<T>(0); N];
    let mut stage = vec![sentinel; N + 2];
    let mut output = vec![sentinel; N + 3];
    let rows = oracle::goldens::<T>(request.range_policy, dead);
    assert!(!rows.is_empty());
    let routes = if cuda.is_some() {
        &["actual_source", "explicit_ir", "independent_sdk"][..]
    } else {
        &["actual_source", "explicit_ir"][..]
    };
    for route in routes {
        for row in &rows {
            inputs.fill(oracle::small(0));
            inputs[7] = row.input;
            stage.fill(sentinel);
            output.fill(sentinel);
            let result = call(
                route,
                &mut prepared,
                &mut native,
                &inputs,
                &row.seed,
                &mut stage,
                &mut output,
                dead,
                request.range_policy,
                &mut host,
            );
            assert_eq!(
                result,
                Ok((row.first != 0).then_some((
                    7,
                    row.first,
                    request.range_policy == PcuRangePolicy::Clamp
                )))
            );
            if row.first != 0 && request.range_policy == PcuRangePolicy::Reject {
                oracle::bits(&stage, &vec![sentinel; N + 2]);
                oracle::bits(&output, &vec![sentinel; N + 3]);
            } else {
                if !dead {
                    let mut expected = vec![row.seed; N];
                    expected[7] = row.stage;
                    oracle::bits(&stage[..N], &expected);
                }
                let mut expected = vec![oracle::small(0); N];
                expected[7] = row.output;
                oracle::bits(&output[..N], &expected);
                oracle::bits(&stage[N..], &[sentinel; 2]);
                oracle::bits(&output[N..], &[sentinel; 3]);
            }
            inputs.fill(oracle::small(0));
            let seed = oracle::small::<T>(1);
            assert_eq!(
                call(
                    route,
                    &mut prepared,
                    &mut native,
                    &inputs,
                    &seed,
                    &mut stage,
                    &mut output,
                    dead,
                    request.range_policy,
                    &mut host
                ),
                Ok(None)
            );
            oracle::bits(&output[..N], &vec![oracle::small(0); N]);
            if !dead {
                oracle::bits(&stage[..N], &vec![seed; N]);
            }
            let old_stage = stage.clone();
            let old_output = output.clone();
            assert_eq!(
                call(
                    route,
                    &mut prepared,
                    &mut native,
                    &inputs[..N - 1],
                    &seed,
                    &mut stage,
                    &mut output,
                    dead,
                    request.range_policy,
                    &mut host
                ),
                Err(())
            );
            oracle::bits(&stage, &old_stage);
            oracle::bits(&output, &old_output);
        }
        // Sibling shape failures must preserve the complete host transaction before any publication.
        let old_stage = stage.clone();
        let old_output = output.clone();
        inputs.fill(oracle::small(0));
        let seed = oracle::small::<T>(1);
        if !dead {
            assert_eq!(
                call(
                    route,
                    &mut prepared,
                    &mut native,
                    &inputs,
                    &seed,
                    &mut stage[..N - 1],
                    &mut output,
                    dead,
                    request.range_policy,
                    &mut host
                ),
                Err(())
            );
            oracle::bits(&stage, &old_stage);
            oracle::bits(&output, &old_output);
        }
        assert_eq!(
            call(
                route,
                &mut prepared,
                &mut native,
                &inputs,
                &seed,
                &mut stage,
                &mut output[..N - 1],
                dead,
                request.range_policy,
                &mut host
            ),
            Err(())
        );
        oracle::bits(&stage, &old_stage);
        oracle::bits(&output, &old_output);
        // Two simultaneous faults prove earliest invocation, retaining the first operation at that lane.
        inputs.fill(oracle::small(0));
        inputs[3] = oracle::limit::<T>(true);
        inputs[7] = inputs[3];
        let seed = if T::SIGNED {
            oracle::limit::<T>(true)
        } else {
            oracle::small(1)
        };
        if T::SIGNED {
            let (_, _, fault) = oracle::law(inputs[3], seed, dead);
            stage.fill(sentinel);
            output.fill(sentinel);
            assert_eq!(
                call(
                    route,
                    &mut prepared,
                    &mut native,
                    &inputs,
                    &seed,
                    &mut stage,
                    &mut output,
                    dead,
                    request.range_policy,
                    &mut host
                ),
                Ok(Some((
                    3,
                    fault,
                    request.range_policy == PcuRangePolicy::Clamp
                )))
            );
        }
    }
    let zero = oracle::small::<T>(0);
    let one = oracle::small::<T>(1);
    let banks = [vec![zero; N], vec![one; N]];
    let seeds = [
        oracle::limit::<T>(true).pcu_checked_add(one).unwrap(),
        oracle::limit::<T>(false)
            .pcu_checked_sub(oracle::small(2))
            .unwrap(),
    ];
    let wanted = std::array::from_fn::<_, 2, _>(|b| {
        let (a, o, f) = oracle::law(banks[b][0], seeds[b], dead);
        assert_eq!(f, 0);
        (vec![a; N], vec![o; N])
    });
    let profile = format!(
        "{:?}/{}/{}/{}/{:?}/{}",
        request.numerical_mode,
        T::LABEL,
        N,
        u8::from(grid),
        request.range_policy,
        if dead { "dead" } else { "staged" }
    );
    let mut group = criterion.benchmark_group(format!("cuda_composed_integer/{profile}"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for route in routes {
        if !semantics && cuda.is_some() {
            super::activity::activity_guard();
        }
        let mut bank = 0;
        let mut warm = || {
            bank ^= 1;
            let started = Instant::now();
            let result = call(
                route,
                &mut prepared,
                &mut native,
                &banks[bank],
                &seeds[bank],
                &mut stage,
                &mut output,
                dead,
                request.range_policy,
                &mut host,
            );
            let elapsed = started.elapsed();
            assert_eq!(result, Ok(None));
            if !dead {
                oracle::bits(&stage[..N], &wanted[bank].0);
            }
            oracle::bits(&output[..N], &wanted[bank].1);
            oracle::bits(&stage[N..], &[sentinel; 2]);
            oracle::bits(&output[N..], &[sentinel; 3]);
            elapsed
        };
        let _ = warm();
        let scores = SCORES.load(Ordering::Relaxed);
        if semantics {
            for _ in 0..3 {
                let _ = warm();
            }
            println!(
                "composed-semantic/{profile}/{route}: fixed goldens, first fault, full prefixes, rollback/Clamp/tails/retry/shape PASS"
            );
        }
        #[cfg(feature = "allocation-census")]
        if cuda.is_some() {
            let counter = &direct_counter;
            let direct_before = counter.as_ref().unwrap().get();
            let before = fusion_pcu_cuda::cuda_api_census();
            let ((), counts) = super::allocations::measure(|| {
                for _ in 0..64 {
                    std::hint::black_box(warm());
                }
            });
            let after = fusion_pcu_cuda::cuda_api_census();
            assert_eq!(after.symbol_resolutions, before.symbol_resolutions);
            assert_eq!(after.module_loads, before.module_loads);
            assert_eq!(after.allocations, before.allocations);
            assert_eq!(after.frees, before.frees);
            println!(
                "composed-census/{profile}/{route}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; wrapped-before={before:?}; wrapped-after={after:?}; direct-SDK={:?}",
                counts.alloc_calls,
                counts.realloc_calls,
                counts.dealloc_calls,
                counts.requested_bytes,
                counter.as_ref().unwrap().get().delta(direct_before)
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        if !semantics {
            group.bench_function(*route, |b| {
                b.iter_custom(|iterations| (0..iterations).map(|_| warm()).sum::<Duration>());
            });
        }
        assert_eq!(
            SCORES.load(Ordering::Relaxed),
            scores,
            "warm source reranked"
        );
    }
    group.finish();
    if let Some(native) = native.filter(|_| {
        T::LABEL == "u8"
            && N == 65
            && !grid
            && !dead
            && request.numerical_mode == PcuNumericalMode::Boundary
            && request.range_policy == PcuRangePolicy::Reject
            && semantics
    }) {
        native.known_terminal_retirement_witness();
    }
}
fn width<T: Format, const N: usize, B: PcuHostKernelBackend>(
    criterion: &mut Criterion,
    backend: &B,
    cuda: Option<&CudaOwnedDispatchBackend>,
    request: PcuImplementationRequirements,
    semantics: bool,
) where
    B::Error: std::fmt::Debug,
    <B::Prepared as PcuPreparedHostKernel>::Error: FaultError,
{
    let representative_timing =
        !semantics && std::env::var_os("PCU_COMPOSED_INTEGER_REPRESENTATIVE_TIMING").is_some();
    if representative_timing
        && (!(T::LABEL == "u32" || T::LABEL == "u512")
            || request.numerical_mode != PcuNumericalMode::Boundary
            || request.range_policy != PcuRangePolicy::Reject)
    {
        return;
    }
    for dead in [false, true] {
        for grid in [false, true] {
            if representative_timing && grid {
                continue;
            }
            let bindings = if dead {
                source::dead_direct_bindings::<T>().to_vec()
            } else {
                source::direct_bindings::<T>().to_vec()
            };
            let mut body = ir::body(T::TYPE, grid, dead, request);
            let grid_ops = [
                PcuDispatchOp::GridStrideLoop {
                    extent: u32::try_from(N).unwrap(),
                    body: &body,
                },
                ir::RETURN,
            ];
            let direct_ops = {
                let mut v = body.clone();
                v.push(ir::RETURN);
                v
            };
            let explicit = ir::kernel(
                T::TYPE,
                if grid { 3 } else { u32::try_from(N).unwrap() },
                &bindings,
                if grid { &grid_ops } else { &direct_ops },
                request,
            );
            let mut check = |source_ir: &PcuDispatchKernelIr<'_>| {
                assert_eq!(source_ir.numerical_requirements, request);
                case::<T, N, B>(
                    criterion,
                    backend,
                    cuda,
                    source_ir,
                    &explicit,
                    grid,
                    dead,
                    semantics,
                    |input, seed, stage, output| match (dead, grid) {
                        (false, false) => source::direct::<T, N>(input, seed, stage, output),
                        (false, true) => source::grid::<T, N>(input, seed, stage, output),
                        (true, false) => source::dead_direct::<T, N>(input, seed, output),
                        (true, true) => source::dead_grid::<T, N>(input, seed, output),
                    },
                );
            };
            match (dead, grid) {
                (false, false) => source::__direct_ir_with_float_underflow_policy::<T, N>(
                    &bindings,
                    request.float_underflow,
                    request.range_policy,
                    request,
                )
                .unwrap()
                .with_ir(&mut check),
                (false, true) => source::__grid_ir_with_float_underflow_policy::<T, N>(
                    &bindings,
                    request.float_underflow,
                    request.range_policy,
                    request,
                )
                .unwrap()
                .with_ir(&mut check),
                (true, false) => source::__dead_direct_ir_with_float_underflow_policy::<T, N>(
                    &bindings,
                    request.float_underflow,
                    request.range_policy,
                    request,
                )
                .unwrap()
                .with_ir(&mut check),
                (true, true) => source::__dead_grid_ir_with_float_underflow_policy::<T, N>(
                    &bindings,
                    request.float_underflow,
                    request.range_policy,
                    request,
                )
                .unwrap()
                .with_ir(&mut check),
            }
            body.clear();
        }
    }
}
fn matrix<B: PcuHostKernelBackend>(
    criterion: &mut Criterion,
    backend: &B,
    cuda: Option<&CudaOwnedDispatchBackend>,
    semantics: bool,
) where
    B::Error: std::fmt::Debug,
    <B::Prepared as PcuPreparedHostKernel>::Error: FaultError,
{
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let request = PcuImplementationRequirements {
                numerical_mode,
                range_policy,
                float_underflow: PcuFloatUnderflowPolicy::IeeeAfterRounding,
                numerical_options: PcuNumericalOptions {
                    compound_arithmetic: PcuCompoundArithmeticPolicy::Checked,
                    precision: PcuPrecisionPolicy::Preserve,
                    reproducibility: PcuReproducibility::PortableV1,
                },
            };
            global::configure(global::PcuExecutionPolicy {
                backend: if cuda.is_some() {
                    global::PcuBackendChoice::Cuda
                } else {
                    global::PcuBackendChoice::Cpu
                },
                device: cuda.map(|b| b.device_identity().device_id()),
                numerical_mode,
                range_policy,
                float_underflow: request.float_underflow,
                numerical_options: request.numerical_options,
                score_invocation: Some(score),
                ..Default::default()
            })
            .unwrap();
            macro_rules! widths {($($ty:ty),+)=>{$(width::<$ty,65,B>(criterion,backend,cuda,request,semantics);width::<$ty,4096,B>(criterion,backend,cuda,request,semantics);)+};}
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
                fusion_pcu::PcuU256,
                fusion_pcu::PcuI256,
                fusion_pcu::PcuU512,
                fusion_pcu::PcuI512
            );
            global::clear_thread_cache().unwrap();
        }
    }
}
pub fn run(criterion: &mut Criterion) {
    if std::env::var_os("PCU_COMPOSED_INTEGER_CPU_REFERENCE").is_some() {
        matrix(
            criterion,
            &fusion_pcu::cpu::PcuCpuHostBackend::scalar(),
            None,
            true,
        );
    } else {
        let semantics = std::env::var_os("PCU_COMPOSED_INTEGER_SEMANTICS").is_some();
        if !semantics {
            super::activity::activity_guard();
        }
        let (_, backend, _) = super::selection::selected_device();
        matrix(
            criterion,
            backend.as_ref(),
            Some(backend.as_ref()),
            semantics,
        );
    }
}

//! Producer storage is distinct from the closest source's explicit dense input banks.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCompoundArithmeticPolicy,
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuOwnedDispatchBackend,
    PcuPrecisionPolicy,
    PcuExecutionFault,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    Tensor,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorElement,
    TensorPointwiseGroupingPolicy,
};
use fusion_pcu_rocm::{RocmOwnedDispatchBackend, RocmOwnedTensorAssessor, RocmTensorExecutionError};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use std::time::Duration;
use std::{rc::Rc, time::Instant};
use super::{
    native::Native,
    oracle::{self, Format},
    source,
};
type Status = Result<Option<(u32, u32, bool)>, ()>;
fn fault_status(fault: Option<PcuExecutionFault>) -> Status {
    fault.map_or(Err(()), |f| {
        Ok(Some((
            u32::try_from(f.invocation_id).unwrap(),
            oracle::code(f.kind),
            f.recovered,
        )))
    })
}
fn tensor_fault(error: &RocmTensorExecutionError) -> Status {
    match error {
        RocmTensorExecutionError::ExecutionFault(fault) => fault_status(Some(*fault)),
        _ => Err(()),
    }
}
fn expected<T: Format>(input: &[T], constant: &[T], uniform: &[T]) -> Vec<T> {
    input
        .iter()
        .zip(constant)
        .zip(uniform)
        .map(|((&x, &c), &u)| x.pcu_checked_add(c).unwrap().pcu_checked_mul(u).unwrap())
        .collect()
}
fn verify<T: Format>(actual: &[T], want: &[T], sentinel: T) {
    oracle::bits(&actual[..want.len()], want);
    oracle::bits(&actual[want.len()..], &[sentinel; 2]);
}
fn high<T: Format>(lane: usize) -> T {
    let mut bytes = [0_u8; 64];
    bytes[0] = u8::try_from(lane % 3 + 1).unwrap();
    bytes[T::ENCODED_SIZE - 1] |= 8;
    T::from_bytes(&bytes[..T::ENCODED_SIZE])
}
#[allow(clippy::too_many_lines)] // Exact producer/source/SDK disposition and owner lifecycle share one profile.
fn case<T: Format + TensorElement, const N: usize>(
    c: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    options: PcuNumericalOptions,
    mode: PcuNumericalMode,
    semantics: bool,
) {
    #[cfg(feature = "allocation-census")]
    let _ = (&mut *c, semantics);
    let pool = PcuMemoryPoolId(0x4954_4c49);
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(backend.device_identity().device_id()),
        numerical_mode: mode,
        numerical_options: options,
        ..Default::default()
    })
    .unwrap();
    let root = RocmOwnedTensorAssessor::new(Rc::clone(backend)).unwrap();
    let assessor = root.assessor();
    let constant: Vec<T> = (0..N).map(high).collect();
    let uniform = vec![oracle::small::<T>(2); N];
    let banks = [
        vec![oracle::small::<T>(0); N],
        vec![oracle::small::<T>(1); N],
    ];
    let want = [
        expected(&banks[0], &constant, &uniform),
        expected(&banks[1], &constant, &uniform),
    ];
    let sentinel = oracle::small::<T>(77);
    let mut observed = vec![sentinel; N + 2];
    let mut graph = Graph::default();
    graph.set_numerical_mode(mode);
    graph.set_numerical_options(options);
    let x = graph.input_typed::<T>([N]).unwrap();
    let literal = graph.constant_typed(Tensor::new([N], constant.clone()).unwrap());
    let factor = graph.uniform_typed([N], uniform[0]).unwrap();
    let sum = graph.add_typed(x, literal).unwrap();
    let product = graph.mul_typed(sum, factor).unwrap();
    let program = graph
        .into_selected_program(
            &[product.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let prepared = assessor.prepare_owned_program(program).unwrap();
    let mut native = Native::new::<T, N>(backend.device_identity().device_id());
    let mut memory = backend.memory_provider(pool);
    // Literal transport and escaping storage get their own selected-output schedule.
    let mut literal_graph = Graph::default();
    literal_graph.set_numerical_mode(mode);
    literal_graph.set_numerical_options(options);
    let l = literal_graph.constant_typed(Tensor::new([N], constant.clone()).unwrap());
    let u = literal_graph.uniform_typed([N], uniform[0]).unwrap();
    let literal_program = literal_graph
        .into_selected_program(
            &[l.erase(), u.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let literals = assessor.prepare_owned_program(literal_program).unwrap();
    let mut escaped = assessor
        .execute_owned_program_outputs::<T, _>(&literals, &[], pool, &mut memory)
        .unwrap();
    for ((_, owner), want) in escaped.iter().zip([&constant, &uniform]) {
        backend
            .download_buffer(pool, owner.buffer(), &mut observed[..N])
            .unwrap();
        verify(&observed, want, sentinel);
    }
    let mutate: Vec<u8> = std::iter::repeat_n(oracle::small::<T>(3).encode_le(), N)
        .flat_map(|v| v.as_ref().to_vec())
        .collect();
    memory
        .transfer_to(escaped[0].1.resource_mut(), 0, &mutate)
        .unwrap();
    let original = assessor
        .execute_owned_program_outputs::<T, _>(&literals, &[], pool, &mut memory)
        .unwrap();
    backend
        .download_buffer(pool, original[0].1.buffer(), &mut observed[..N])
        .unwrap();
    verify(&observed, &constant, sentinel);
    drop(original);
    let mut consumer_graph = Graph::default();
    consumer_graph.set_numerical_mode(mode);
    consumer_graph.set_numerical_options(options);
    let consumer_input = consumer_graph.input_typed::<T>([N]).unwrap();
    let identity = consumer_graph.identity_typed(consumer_input).unwrap();
    let identity_plan = assessor
        .prepare_owned_program(
            consumer_graph
                .into_selected_program(
                    &[identity.erase()],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .unwrap(),
        )
        .unwrap();
    let consumed = assessor
        .execute_owned_program_output_from_inputs::<T, _>(
            &identity_plan,
            &[(
                consumer_input.erase(),
                &assessor
                    .borrow_device_input_ref(&escaped[0].1, pool)
                    .unwrap(),
            )],
            pool,
            &mut memory,
        )
        .unwrap();
    backend
        .download_buffer(pool, consumed.buffer(), &mut observed[..N])
        .unwrap();
    verify(&observed, &vec![oracle::small::<T>(3); N], sentinel);
    drop(consumed);
    drop(identity_plan);
    drop(literals);
    // Full selected constants survive plan/cache drop independently of returned mutable storage.
    global::clear_thread_cache().unwrap();
    backend
        .download_buffer(pool, escaped[1].1.buffer(), &mut observed[..N])
        .unwrap();
    verify(&observed, &uniform, sentinel);
    drop(escaped);
    let source_run = |input: &[T], out: &mut [T]| match source::pipeline(input, &constant, &uniform)
    {
        Ok(owner) => {
            owner.read_into(out).map_err(|_| ())?;
            Ok(None)
        }
        Err(error) => fault_status(error.arithmetic_fault()),
    };
    let mut graph_run = |input: &[T], out: &mut [T]| {
        if input.len() != N || out.len() < N {
            return Err(());
        }
        let input = PcuDeviceTensor::new([N], backend.upload_buffer(pool, input).unwrap()).unwrap();
        match assessor.execute_owned_program_output_from_inputs::<T, _>(
            &prepared,
            &[(
                x.erase(),
                &assessor.borrow_device_input_ref(&input, pool).unwrap(),
            )],
            pool,
            &mut memory,
        ) {
            Ok(owner) => {
                backend
                    .download_buffer(pool, owner.buffer(), &mut out[..N])
                    .unwrap();
                Ok(None)
            }
            Err(error) => tensor_fault(&error),
        }
    };
    #[cfg(feature = "allocation-census")]
    let native_counter = native.counter();
    for route in [
        "closest_source_dense_banks",
        "explicit_graph_cold_producers",
        "independent_sdk_dense_banks",
    ] {
        let mut call = |input: &[T], out: &mut [T]| match route {
            "closest_source_dense_banks" => source_run(input, out),
            "explicit_graph_cold_producers" => graph_run(input, out),
            _ => native
                .call(input, &constant, &uniform, out)
                .map(|fault| fault.map(|(index, code)| (index, code, false))),
        };
        for phase in 0..2 {
            #[cfg(feature = "allocation-census")]
            let before = fusion_pcu_rocm::rocm_api_census();
            assert_eq!(call(&banks[phase], &mut observed), Ok(None));
            verify(&observed, &want[phase], sentinel);
            #[cfg(feature = "allocation-census")]
            if route == "closest_source_dense_banks" {
                let after = fusion_pcu_rocm::rocm_api_census();
                let guarded = after.guarded_chain_submissions - before.guarded_chain_submissions;
                assert!(guarded <= 1);
                assert_eq!(
                    after.async_host_to_device_copies - before.async_host_to_device_copies,
                    if phase == 0 { guarded } else { 3 + guarded },
                );
            }
        }
        #[cfg(feature = "allocation-census")]
        if route == "closest_source_dense_banks" {
            let before = fusion_pcu_rocm::rocm_api_census();
            let resident = source::identity(&banks[0]).unwrap();
            let identity = fusion_pcu_rocm::rocm_api_census();
            assert_eq!(
                identity.async_host_to_device_copies,
                before.async_host_to_device_copies
            );
            let mixed = source::mixed(&resident, &constant, &uniform).unwrap();
            let after = fusion_pcu_rocm::rocm_api_census();
            assert_eq!(
                after.async_host_to_device_copies - identity.async_host_to_device_copies,
                after.guarded_chain_submissions - identity.guarded_chain_submissions
            );
            mixed.read_into(&mut observed).unwrap();
            verify(&observed, &want[0], sentinel);
            // Restore the most recent successful source output used by the rollback witness.
            assert_eq!(call(&banks[1], &mut observed), Ok(None));
        }
        let previous = observed.clone();
        let mut bad = banks[0].clone();
        bad[7] = oracle::limit::<T>(false);
        assert_eq!(call(&bad, &mut observed), Ok(Some((7, 3, false))));
        oracle::bits(&observed, &previous);
        let mut half = [255_u8; 64];
        half[T::ENCODED_SIZE - 1] = if T::SIGNED { 0x3f } else { 0x7f };
        bad[7] = T::from_bytes(&half[..T::ENCODED_SIZE]);
        assert_eq!(call(&bad, &mut observed), Ok(Some((7, 3, false))));
        oracle::bits(&observed, &previous);
        // Graph-wide Add failure wins over an earlier invocation's later Mul failure.
        bad[3] = bad[7];
        assert_eq!(call(&bad, &mut observed), Ok(Some((3, 3, false))));
        oracle::bits(&observed, &previous);
        bad[7] = oracle::limit::<T>(false);
        assert_eq!(call(&bad, &mut observed), Ok(Some((7, 3, false))));
        oracle::bits(&observed, &previous);
        bad[3] = oracle::limit::<T>(false);
        bad[7] = T::from_bytes(&half[..T::ENCODED_SIZE]);
        assert_eq!(call(&bad, &mut observed), Ok(Some((3, 3, false))));
        oracle::bits(&observed, &previous);
        assert_eq!(call(&banks[1], &mut observed), Ok(None));
        verify(&observed, &want[1], sentinel);
        let before_short = observed.clone();
        assert_eq!(call(&banks[0][..N - 1], &mut observed), Err(()));
        oracle::bits(&observed, &before_short);
        assert_eq!(call(&banks[0], &mut observed[..N - 1]), Err(()));
        oracle::bits(&observed, &before_short);
        assert_eq!(call(&banks[0], &mut observed), Ok(None));
        verify(&observed, &want[0], sentinel);
        let profile = format!(
            "{mode:?}/{:?}/{:?}/{}/{N}/{route}",
            options.compound_arithmetic,
            options.precision,
            T::LABEL
        );
        println!(
            "integer-literal-semantic/{profile}: exact high-bit producers/escaped mutation/replay/fatal rollback/retry/tails PASS"
        );
        let mut phase = 0;
        let mut warm = || {
            phase ^= 1;
            let start = Instant::now();
            assert_eq!(call(&banks[phase], &mut observed), Ok(None));
            let elapsed = start.elapsed();
            verify(&observed, &want[phase], sentinel);
            elapsed
        };
        let _ = warm();
        #[cfg(feature = "allocation-census")]
        {
            let before = fusion_pcu_rocm::rocm_api_census();
            let sdk_before = native_counter.get();
            let ((), counts) = super::allocations::measure(|| {
                for _ in 0..64 {
                    std::hint::black_box(warm());
                }
            });
            let after = fusion_pcu_rocm::rocm_api_census();
            let sdk = native_counter.get().delta(sdk_before);
            if route == "closest_source_dense_banks" {
                // Genuine all-host #[pcu] execution: either two host-observed checked stages,
                // or a proved guarded scope with one status-slab reset/readback and final wait.
                let guarded = after.guarded_chain_submissions - before.guarded_chain_submissions;
                assert!(guarded == 0 || guarded == 64);
                if std::env::var_os("PCU_GUARDED_CHAIN_WITNESS").is_some() {
                    assert_eq!(
                        guarded, 64,
                        "source witness must exercise guarded execution"
                    );
                }
                assert_eq!(
                    after.guarded_kernel_launches - before.guarded_kernel_launches,
                    2 * guarded
                );
                assert_eq!(
                    after.host_to_device_copies - before.host_to_device_copies,
                    192 + guarded
                );
                assert_eq!(
                    after.device_to_host_copies - before.device_to_host_copies,
                    192 - guarded
                );
                assert_eq!(after.kernel_launches - before.kernel_launches, 128);
                assert_eq!(
                    after.event_creates - before.event_creates,
                    128 - 2 * guarded
                );
                assert_eq!(after.event_records - before.event_records, 128 - guarded);
                assert_eq!(after.event_waits - before.event_waits, 128 - guarded);
                assert_eq!(
                    after.event_destroys - before.event_destroys,
                    128 - 2 * guarded
                );
                assert_eq!(counts.alloc_calls, 64);
                assert_eq!(counts.realloc_calls, 0);
                assert_eq!(
                    after.stream_synchronizations,
                    before.stream_synchronizations
                );
                assert_eq!(
                    after.device_synchronizations,
                    before.device_synchronizations
                );
                assert_eq!(after.module_loads, before.module_loads);
                assert_eq!(after.symbol_resolutions, before.symbol_resolutions);
            }
            if route == "independent_sdk_dense_banks" {
                assert_eq!(
                    (
                        sdk.uploads,
                        sdk.downloads,
                        sdk.resets,
                        sdk.launches,
                        sdk.event_records,
                        sdk.event_waits
                    ),
                    (192, 128, 64, 128, 64, 64)
                );
                assert_eq!(
                    sdk.uploaded_bytes,
                    u64::try_from(192 * N * T::ENCODED_SIZE).unwrap()
                );
                assert_eq!(
                    sdk.downloaded_bytes,
                    u64::try_from(64 * (N * T::ENCODED_SIZE + 16)).unwrap()
                );
                assert_eq!(
                    (
                        counts.alloc_calls,
                        counts.realloc_calls,
                        counts.dealloc_calls,
                        counts.requested_bytes
                    ),
                    (0, 0, 0, 0)
                );
            }
            println!(
                "integer-literal-census/{profile}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; before={before:?}; after={after:?}; sdk={sdk:?}",
                counts.alloc_calls,
                counts.realloc_calls,
                counts.dealloc_calls,
                counts.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        if !semantics {
            super::activity::activity_guard();
            let mut group = c.benchmark_group(format!("rocm_integer_literal_support/{profile}"));
            group.sample_size(20);
            group.warm_up_time(Duration::from_millis(500));
            group.measurement_time(Duration::from_secs(2));
            group.bench_function("full_host_boundary", |b| {
                b.iter_custom(|n| (0..n).map(|_| warm()).sum::<Duration>());
            });
            group.finish();
        }
    }
}
fn cpu_reference<T: Format + TensorElement, const N: usize>(
    options: PcuNumericalOptions,
    mode: PcuNumericalMode,
) {
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_mode: mode,
        numerical_options: options,
        ..Default::default()
    })
    .unwrap();
    let constant: Vec<T> = (0..N).map(high).collect();
    let uniform = vec![oracle::small::<T>(2); N];
    let input = vec![oracle::small::<T>(1); N];
    let want = expected(&input, &constant, &uniform);
    let sentinel = oracle::small::<T>(77);
    let mut observed = vec![sentinel; N + 2];
    source::pipeline(&input, &constant, &uniform)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    verify(&observed, &want, sentinel);
    let mut bad = input;
    bad[7] = oracle::limit::<T>(false);
    let Err(error) = source::pipeline(&bad, &constant, &uniform) else {
        panic!("missing CPU numeric fault");
    };
    assert_eq!(
        fault_status(error.arithmetic_fault()),
        Ok(Some((7, 3, false)))
    );
    let mut half = [255_u8; 64];
    half[T::ENCODED_SIZE - 1] = if T::SIGNED { 0x3f } else { 0x7f };
    bad[3] = T::from_bytes(&half[..T::ENCODED_SIZE]);
    let Err(error) = source::pipeline(&bad, &constant, &uniform) else {
        panic!("missing CPU mixed-stage fault");
    };
    assert_eq!(
        fault_status(error.arithmetic_fault()),
        Ok(Some((7, 3, false)))
    );
    bad[3] = oracle::limit::<T>(false);
    bad[7] = T::from_bytes(&half[..T::ENCODED_SIZE]);
    let Err(error) = source::pipeline(&bad, &constant, &uniform) else {
        panic!("missing CPU reverse mixed-stage fault");
    };
    assert_eq!(
        fault_status(error.arithmetic_fault()),
        Ok(Some((3, 3, false)))
    );
    println!(
        "integer-literal-cpu-reference/{mode:?}/{:?}/{:?}/{}/{N}: closest dense-bank source PASS",
        options.compound_arithmetic,
        options.precision,
        T::LABEL
    );
}
pub fn run(c: &mut Criterion) {
    let semantics = std::env::var_os("PCU_INTEGER_TENSOR_LITERAL_SEMANTICS").is_some()
        || std::env::args().any(|x| x == "--test");
    if !semantics {
        super::activity::activity_guard();
    }
    let reference = std::env::var_os("PCU_INTEGER_TENSOR_LITERAL_CPU_REFERENCE").is_some();
    let backend = (!reference).then(|| super::selection::selected_device().1);
    if std::env::var_os("PCU_PHYSICAL_WORK_WITNESS").is_some() {
        super::physical_work::run(
            c,
            backend.as_ref().unwrap().device_identity().device_id(),
            semantics,
        );
        return;
    }
    #[cfg(not(feature = "allocation-census"))]
    if std::env::var_os("PCU_GUARDED_FAULT_WITNESS").is_some() {
        super::fault_latency::run(c, backend.as_ref().unwrap().device_identity().device_id());
        return;
    }
    // Bounded source/graph/independent-SDK qualification for the queued-upload route.
    // The full format/policy matrix remains the default benchmark behavior.
    if std::env::var_os("PCU_ROCM_QUEUED_UPLOAD_WITNESS").is_some()
        || std::env::var_os("PCU_GUARDED_CHAIN_WITNESS").is_some()
    {
        let backend = backend
            .as_ref()
            .expect("queued-upload witness requires ROCm");
        let options = PcuNumericalOptions::default();
        case::<u32, 65>(c, backend, options, PcuNumericalMode::Strict, semantics);
        case::<u32, 4096>(c, backend, options, PcuNumericalMode::Strict, semantics);
        return;
    }
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                let options = PcuNumericalOptions {
                    compound_arithmetic,
                    precision,
                    ..Default::default()
                };
                macro_rules! widths {($($ty:ty),+)=>{$(if let Some(backend) = &backend {
                    case::<$ty,65>(c,backend,options,mode,semantics); case::<$ty,4096>(c,backend,options,mode,semantics);
                } else { cpu_reference::<$ty,65>(options,mode); cpu_reference::<$ty,4096>(options,mode); })+};}
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
            }
        }
    }
}

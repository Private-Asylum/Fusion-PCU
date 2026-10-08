//! Genuine source and explicit graph freeze the same typed immutable producers.
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
    PcuFloatUnderflowPolicy,
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
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmOwnedDispatchBackend,
    RocmOwnedTensorAssessor,
    RocmTensorExecutionError,
};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use std::time::Duration;
#[rustfmt::skip]
use std::{
    rc::Rc,
    time::Instant,
};
#[rustfmt::skip]
use super::{
    native::Native,
    oracle::{
        self,
        Format,
    },
    source::{
        self,
        ProducerSource,
    },
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
fn expected<T: Format>(
    input: &[T],
    constant: &[T],
    uniform: &[T],
    policy: PcuFloatUnderflowPolicy,
) -> Vec<T> {
    oracle::pipeline(input, constant, uniform, policy).unwrap()
}
fn verify<T: Format>(actual: &[T], want: &[T], sentinel: T) {
    oracle::bits(&actual[..want.len()], want);
    oracle::bits(&actual[want.len()..], &[sentinel; 2]);
}
fn payload<T: Format>(lane: usize) -> T {
    oracle::payload(lane)
}
#[allow(clippy::too_many_lines)] // Exact producer/source/SDK disposition and owner lifecycle share one profile.
fn case<T: ProducerSource + TensorElement, const N: usize, const HALF: bool>(
    c: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    options: PcuNumericalOptions,
    mode: PcuNumericalMode,
    semantics: bool,
    policy: PcuFloatUnderflowPolicy,
) {
    // Optional exact profile substring selects setup as well as timing; every selected case
    // still runs its complete independent oracle, ownership, fault, retry and tail checks.
    if let Ok(filter) = std::env::var("PCU_LOW_TENSOR_PRODUCER_CASE_FILTER") {
        let profile = format!(
            "{mode:?}/{:?}/{:?}/{policy:?}/half={HALF}/{}/{N}",
            options.compound_arithmetic,
            options.precision,
            T::LABEL,
        );
        if !profile.contains(&filter) {
            return;
        }
    }
    #[cfg(feature = "allocation-census")]
    let _ = (&mut *c, semantics);
    verify_capture::<T, N, HALF>(mode, options, policy);
    let pool = PcuMemoryPoolId(0x4954_4c49);
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(backend.device_identity().device_id()),
        numerical_mode: mode,
        numerical_options: options,
        float_underflow: policy,
        ..Default::default()
    })
    .unwrap();
    let root = RocmOwnedTensorAssessor::new(Rc::clone(backend)).unwrap();
    let assessor = root.assessor();
    let constant: Vec<T> = (0..N).map(payload).collect();
    let uniform = vec![oracle::factor::<T>(HALF); N];
    let banks = [
        vec![oracle::small::<T>(1); N],
        vec![oracle::small::<T>(2); N],
    ];
    let want = [
        expected(&banks[0], &constant, &uniform, policy),
        expected(&banks[1], &constant, &uniform, policy),
    ];
    let sentinel = T::sentinel();
    let mut observed = vec![sentinel; N + 2];
    let mut graph = Graph::default();
    graph.set_numerical_mode(mode);
    graph.set_numerical_options(options);
    let x = graph.input_typed::<T>([N]).unwrap();
    let literal = graph.constant_typed(Tensor::new([N], constant.clone()).unwrap());
    let factor = graph.uniform_typed([N], uniform[0]).unwrap();
    let sum = graph.add_typed(x, literal).unwrap();
    let product = graph.mul_typed(sum, factor).unwrap();
    for value in [sum.erase(), product.erase()] {
        graph
            .set_value_float_underflow_policy(value, policy)
            .unwrap();
    }
    let program = graph
        .into_selected_program(
            &[product.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let prepared = assessor.prepare_owned_program(program).unwrap();
    let mut native = Native::new::<T, N>(backend.device_identity().device_id(), policy);
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
    // Ordinary literal-only source has no device data inputs. Mutation cannot alias its cached template.
    let mut source_literal = T::literal::<N>(&[]).unwrap();
    source_literal.read_into(&mut observed).unwrap();
    verify(&observed, &constant, sentinel);
    source::overwrite::<T, N>(&[oracle::small::<T>(3)], &mut source_literal).unwrap();
    let consumed_source = source::consume(source_literal).unwrap();
    consumed_source.read_into(&mut observed).unwrap();
    verify(&observed, &vec![oracle::small::<T>(3); N], sentinel);
    let replay = T::literal::<N>(&[]).unwrap();
    replay.read_into(&mut observed).unwrap();
    verify(&observed, &constant, sentinel);
    T::uniform::<HALF>(&banks[0])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    verify(&observed, &uniform, sentinel);
    global::clear_thread_cache().unwrap();
    consumed_source.read_into(&mut observed).unwrap();
    verify(&observed, &vec![oracle::small::<T>(3); N], sentinel);
    drop(consumed_source);
    drop(replay);
    // A healthy CPU resident owner used only as a declaration must impose no ROCm affinity.
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let foreign = source::identity(&banks[0]).unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(backend.device_identity().device_id()),
        numerical_mode: mode,
        numerical_options: options,
        float_underflow: policy,
        ..Default::default()
    })
    .unwrap();
    T::unused_owner::<N>(&foreign)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    verify(&observed, &constant, sentinel);
    assert!(matches!(
        source::identity(&foreign),
        Err(fusion_pcu::PcuExecutionError::Argument(
            fusion_pcu::global::PcuArgumentError::UnsupportedResidentBorrow
        ))
    ));
    drop(foreign);
    let source_run = |input: &[T], out: &mut [T]| match T::pipeline::<N, HALF>(input) {
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
        "actual_source_immutable_producers",
        "explicit_graph_cold_producers",
        "independent_sdk_dense_banks",
    ] {
        let mut call = |input: &[T], out: &mut [T]| match route {
            "actual_source_immutable_producers" => source_run(input, out),
            "explicit_graph_cold_producers" => graph_run(input, out),
            _ => native
                .call(input, &constant, &uniform, out)
                .map(|fault| fault.map(|(index, code)| (index, code, false))),
        };
        for phase in 0..2 {
            assert_eq!(call(&banks[phase], &mut observed), Ok(None));
            verify(&observed, &want[phase], sentinel);
        }
        let previous = observed.clone();
        let mut bad = banks[0].clone();
        bad[7] = T::from(T::MAX + 1);
        let mut fixtures = vec![bad.clone()];
        bad[3] = T::from(T::MAX + 1);
        fixtures.push(bad.clone());
        bad[3] = T::from(T::MAX);
        fixtures.push(bad.clone());
        bad[7] = T::from(T::MAX);
        fixtures.push(bad.clone());
        let mut tiny = banks[0].clone();
        tiny[1] = T::zero();
        fixtures.push(tiny.clone());
        tiny[2] = T::from(1 << T::FRACTION);
        fixtures.push(tiny.clone());
        tiny.clone_from(&banks[0]);
        tiny[2] = T::from(1 << T::FRACTION);
        fixtures.push(tiny.clone());
        for input in fixtures {
            let expected = oracle::pipeline(&input, &constant, &uniform, policy);
            assert_eq!(
                call(&input, &mut observed),
                Ok(expected.as_ref().err().copied())
            );
            if let Ok(want) = expected {
                verify(&observed, &want, sentinel);
            } else {
                oracle::bits(&observed, &previous);
            }
            // Restore a known healthy published result before every next fault check.
            assert_eq!(call(&banks[1], &mut observed), Ok(None));
            verify(&observed, &want[1], sentinel);
        }
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
            "{mode:?}/{:?}/{:?}/{policy:?}/half={HALF}/{}/{N}/{route}",
            options.compound_arithmetic,
            options.precision,
            T::LABEL
        );
        println!(
            "low-producer-semantic/{profile}: exact low-format producers/escaped mutation/replay/fatal rollback/retry/tails PASS"
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
            let pcusdk = [
                after.symbol_resolutions - before.symbol_resolutions,
                after.runtime_calls - before.runtime_calls,
                after.device_selections - before.device_selections,
                after.allocations - before.allocations,
                after.frees - before.frees,
                after.host_to_device_copies - before.host_to_device_copies,
                after.device_to_host_copies - before.device_to_host_copies,
                after.kernel_launches - before.kernel_launches,
                after.event_creates - before.event_creates,
                after.event_records - before.event_records,
                after.event_waits - before.event_waits,
                after.event_destroys - before.event_destroys,
                after.module_loads - before.module_loads,
            ];
            assert_eq!(
                pcusdk,
                match route {
                    "actual_source_immutable_producers" =>
                        [0, 2048, 1024, 64, 64, 64, 192, 128, 128, 128, 128, 128, 0],
                    "explicit_graph_cold_producers" =>
                        [0, 2304, 1152, 128, 128, 64, 192, 128, 128, 128, 128, 128, 0],
                    _ => [0; 13],
                }
            );
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
                "low-producer-census/{profile}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; before={before:?}; after={after:?}; sdk={sdk:?}",
                counts.alloc_calls,
                counts.realloc_calls,
                counts.dealloc_calls,
                counts.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        if !semantics {
            super::activity::activity_guard();
            let mut group = c.benchmark_group(format!("rocm_low_producer_support/{profile}"));
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
fn verify_capture<T: ProducerSource + TensorElement, const N: usize, const HALF: bool>(
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
    policy: PcuFloatUnderflowPolicy,
) {
    use fusion_pcu::dialect::tensor::OpDescriptor;
    let captured = T::capture::<N, HALF>(mode, options, policy).unwrap();
    assert_eq!(captured.argument_indices(), [0]);
    assert_eq!(captured.program().input_values().len(), 1);
    let mut producers = 0;
    for node in captured.program().graph().nodes() {
        if !matches!(node.op, OpDescriptor::Input) {
            assert_eq!(node.numerical_options, options);
        }
        match node.op {
            OpDescriptor::Constant(value) => {
                assert_eq!(node.shape, [N]);
                assert_eq!(node.numerical_mode, None);
                assert_eq!(node.float_underflow_policy, None);
                oracle::bits(
                    value.as_typed::<T>().unwrap().data(),
                    &(0..N).map(payload).collect::<Vec<T>>(),
                );
                producers += 1;
            }
            OpDescriptor::Uniform { value } => {
                assert_eq!(node.shape, [N]);
                assert_eq!(node.numerical_mode, None);
                assert_eq!(node.float_underflow_policy, None);
                oracle::bits(
                    &[value.as_typed::<T>().unwrap()],
                    &[oracle::factor::<T>(HALF)],
                );
                producers += 1;
            }
            OpDescriptor::Add { .. } | OpDescriptor::Mul { .. } => {
                // Scalar primitives cover their whole operation domain; mode only distinguishes compounds.
                assert_eq!(node.numerical_mode, None);
                assert_eq!(node.float_underflow_policy, Some(policy));
            }
            _ => {}
        }
    }
    assert_eq!(producers, 2);
}

fn cpu_reference<T: ProducerSource + TensorElement, const N: usize, const HALF: bool>(
    options: PcuNumericalOptions,
    mode: PcuNumericalMode,
    policy: PcuFloatUnderflowPolicy,
) {
    verify_capture::<T, N, HALF>(mode, options, policy);
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_mode: mode,
        numerical_options: options,
        float_underflow: policy,
        ..Default::default()
    })
    .unwrap();
    let constant: Vec<T> = (0..N).map(payload).collect();
    let uniform = vec![oracle::factor::<T>(HALF); N];
    let input = vec![oracle::small::<T>(1); N];
    let want = expected(&input, &constant, &uniform, policy);
    let sentinel = T::sentinel();
    let mut observed = vec![sentinel; N + 2];
    T::pipeline::<N, HALF>(&input)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    verify(&observed, &want, sentinel);
    observed.fill(sentinel);
    let mut literal_owner = T::literal::<N>(&[]).unwrap();
    literal_owner.read_into(&mut observed).unwrap();
    verify(&observed, &constant, sentinel);
    source::overwrite::<T, N>(&[oracle::small::<T>(3)], &mut literal_owner).unwrap();
    let escaped = source::consume(literal_owner).unwrap();
    T::literal::<N>(&[])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    verify(&observed, &constant, sentinel);
    T::uniform::<HALF>(&input)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    verify(&observed, &uniform, sentinel);
    global::clear_thread_cache().unwrap();
    escaped.read_into(&mut observed).unwrap();
    verify(&observed, &vec![oracle::small::<T>(3); N], sentinel);
    let mut bad = input;
    bad[7] = T::from(T::MAX + 1);
    for lane3 in [T::from(T::MAX), T::from(T::MAX + 1)] {
        bad[3] = lane3;
        let expected = oracle::pipeline(&bad, &constant, &uniform, policy).unwrap_err();
        let error = T::pipeline::<N, HALF>(&bad).unwrap_err();
        assert_eq!(fault_status(error.arithmetic_fault()), Ok(Some(expected)));
    }
    println!(
        "low-producer-cpu-reference/{mode:?}/{:?}/{:?}/{policy:?}/half={HALF}/{}/{N}: genuine immutable producers and zero-input escape PASS",
        options.compound_arithmetic,
        options.precision,
        T::LABEL
    );
}
pub fn run(c: &mut Criterion) {
    let semantics = std::env::var_os("PCU_LOW_TENSOR_PRODUCER_SEMANTICS").is_some()
        || std::env::args().any(|x| x == "--test");
    if !semantics {
        super::activity::activity_guard();
    }
    let reference = std::env::var_os("PCU_LOW_TENSOR_PRODUCER_CPU_REFERENCE").is_some();
    let backend = (!reference).then(|| super::selection::selected_device().1);
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
                for policy in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    macro_rules! widths {($($ty:ty),+)=>{$(if let Some(backend) = &backend {
                        case::<$ty,65,false>(c,backend,options,mode,semantics,policy);case::<$ty,4096,false>(c,backend,options,mode,semantics,policy);
                        case::<$ty,65,true>(c,backend,options,mode,semantics,policy);case::<$ty,4096,true>(c,backend,options,mode,semantics,policy);
                    } else {
                        cpu_reference::<$ty,65,false>(options,mode,policy);cpu_reference::<$ty,4096,false>(options,mode,policy);
                        cpu_reference::<$ty,65,true>(options,mode,policy);cpu_reference::<$ty,4096,true>(options,mode,policy);
                    })+};}
                    widths!(
                        fusion_pcu::PcuF16Bits,
                        fusion_pcu::PcuBf16Bits,
                        fusion_pcu::PcuF8E4M3FnBits,
                        fusion_pcu::PcuF8E5M2Bits
                    );
                }
            }
        }
    }
}

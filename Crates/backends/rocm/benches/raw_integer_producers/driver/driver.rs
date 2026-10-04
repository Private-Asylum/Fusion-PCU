//! Raw producer bits and fresh ownership agree before any optional timing.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuOwnedDispatchBackend,
    PcuPrecisionPolicy,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    Tensor,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
    TensorOperationAssessor,
    TensorOperationSupport,
    TensorUnsupportedReason,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmOwnedDispatchBackend,
    RocmOwnedTensorAssessor,
};
use criterion::Criterion;
use std::rc::Rc;
#[cfg(not(feature = "allocation-census"))]
use std::time::Duration;
#[rustfmt::skip]
use super::{
    native::Control,
    source::{
        self,
        RawSource,
    },
};

// Independent fixed limb vectors deliberately duplicate the source payload declaration.
fn expected<T: RawSource, const N: usize>(uniform: bool) -> Vec<u8> {
    let mut patterns = [
        vec![0_u8; T::ENCODED_SIZE],
        vec![0xff; T::ENCODED_SIZE],
        vec![0_u8; T::ENCODED_SIZE],
    ];
    patterns[0][T::ENCODED_SIZE - 1] = 0x80;
    patterns[2][0] = 7;
    (0..N)
        .flat_map(|index| {
            let pattern = if uniform {
                0
            } else if N == 6 && index >= 3 {
                5 - index
            } else {
                index % 3
            };
            patterns[pattern].clone()
        })
        .collect()
}

#[allow(clippy::chunks_exact_to_as_chunks)] // Stable Rust cannot use T::ENCODED_SIZE as a const-generic expression.
fn verify<T: RawSource>(actual: &[T], bytes: &[u8]) {
    let count = bytes.len() / T::ENCODED_SIZE;
    for (value, expected) in actual[..count]
        .iter()
        .zip(bytes.chunks_exact(T::ENCODED_SIZE))
    {
        assert_eq!(value.encode_le().as_ref(), expected);
    }
    for value in &actual[count..] {
        assert_eq!(value.encode_le().as_ref(), T::SENTINEL.encode_le().as_ref());
    }
}
fn source_owner<T: RawSource, const N: usize>(uniform: bool) -> fusion_pcu::PcuTensor<T> {
    if uniform {
        T::uniform::<N>().unwrap()
    } else {
        T::literal::<N>().unwrap()
    }
}
fn source_lifetime<T: RawSource, const N: usize>(uniform: bool, want: &[u8]) {
    let mut output = vec![T::SENTINEL; N + 2];
    let mut alternate = expected::<T, N>(false);
    for index in 0..N {
        let pattern = if N == 6 && index >= 3 {
            5 - index
        } else {
            index % 3
        };
        let offset = index * T::ENCODED_SIZE;
        alternate[offset] = match pattern {
            0 => alternate[offset] | 1,
            1 => 0xfe,
            _ => 8,
        };
    }
    let alternate_owner = T::alternate::<N>().unwrap();
    alternate_owner.read_into(&mut output).unwrap();
    verify(&output, &alternate);
    let mut owner = source_owner::<T, N>(uniform);
    assert_eq!(owner.shape(), if N == 6 { vec![2, 3] } else { vec![N] });
    owner.read_into(&mut output).unwrap();
    verify(&output, want);
    let mut short = vec![T::SENTINEL; N - 1];
    assert!(owner.read_into(&mut short).is_err());
    verify(&short, &[]);
    if N == 6 {
        source::overwrite_matrix(&[[T::SENTINEL; 3]; 2], &mut owner).unwrap();
    } else {
        source::overwrite::<T, N>(&[T::SENTINEL], &mut owner).unwrap();
    }
    let consumed = source::consume(owner).unwrap();
    consumed.read_into(&mut output).unwrap();
    verify(&output, &[]);
    let replay = source_owner::<T, N>(uniform);
    replay.read_into(&mut output).unwrap();
    verify(&output, want);
    global::clear_thread_cache().unwrap();
    replay.read_into(&mut output).unwrap();
    verify(&output, want);
    consumed.read_into(&mut output).unwrap();
    verify(&output, &[]);
    alternate_owner.read_into(&mut output).unwrap();
    verify(&output, &alternate);
}

#[allow(clippy::too_many_lines)] // One closed profile keeps source/graph/native lifetime and census matched.
#[allow(clippy::chunks_exact_to_as_chunks)] // Stable Rust cannot use T::ENCODED_SIZE as a const-generic expression.
fn case<T: RawSource, const N: usize>(
    c: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
    underflow: PcuFloatUnderflowPolicy,
    semantics: bool,
) {
    #[cfg(feature = "allocation-census")]
    let _ = (&mut *c, semantics);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(backend.device_identity().device_id()),
        numerical_mode: mode,
        numerical_options: options,
        float_underflow: underflow,
        ..Default::default()
    })
    .unwrap();
    T::capture::<N>(mode, options, underflow);
    let root = RocmOwnedTensorAssessor::new(Rc::clone(backend)).unwrap();
    let assessor = root.assessor();
    let pool = PcuMemoryPoolId(0x5241_5746);
    let mut memory = backend.memory_provider(pool);
    let sdk = Control::new(i32::try_from(backend.device_identity().device_id()).unwrap());
    for uniform in [false, true] {
        global::clear_thread_cache().unwrap();
        let kind = if uniform { "uniform" } else { "constant" };
        let want = expected::<T, N>(uniform);
        source_lifetime::<T, N>(uniform, &want);
        let values = want
            .chunks_exact(T::ENCODED_SIZE)
            .map(T::from_bytes)
            .collect::<Vec<_>>();
        let mut portable = Graph::default();
        portable.set_numerical_options(fusion_pcu::PcuNumericalOptions {
            reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
            ..options
        });
        let constant = portable.constant_typed(
            Tensor::new(if N == 6 { vec![2, 3] } else { vec![N] }, values.clone()).unwrap(),
        );
        let portable_uniform = portable
            .uniform_typed(if N == 6 { vec![2, 3] } else { vec![N] }, values[0])
            .unwrap();
        for value in [constant.erase(), portable_uniform.erase()] {
            assert!(
                matches!(assessor.assess_node(&portable,portable.node(value).unwrap()),
                TensorOperationSupport::Unsupported {reason:TensorUnsupportedReason::NumericalPolicy {
                    requirement:fusion_pcu::PcuNumericalRequirement::Reproducibility,options:ref refused,
                }} if *refused==portable.node(value).unwrap().numerical_options)
            );
        }
        let mut graph = Graph::default();
        graph.set_numerical_mode(mode);
        graph.set_numerical_options(options);
        let shape = if N == 6 { vec![2, 3] } else { vec![N] };
        let output = if uniform {
            graph.uniform_typed(shape, values[0]).unwrap()
        } else {
            graph.constant_typed(Tensor::new(shape, values).unwrap())
        };
        let prepared = assessor
            .prepare_owned_program(
                graph
                    .into_selected_program(
                        &[output.erase()],
                        TensorArithmeticRewritePolicy::Disabled,
                        TensorArithmeticCapability::Strict,
                        TensorPointwiseGroupingPolicy::Disabled,
                    )
                    .unwrap(),
            )
            .unwrap();
        let mut observed = vec![T::SENTINEL; N + 2];
        let mutate = T::SENTINEL.encode_le().as_ref().repeat(N);
        let mut escaped = assessor
            .execute_owned_program_output_from_inputs::<T, _>(&prepared, &[], pool, &mut memory)
            .unwrap();
        memory
            .transfer_to(escaped.resource_mut(), 0, &mutate)
            .unwrap();
        let replay = assessor
            .execute_owned_program_output_from_inputs::<T, _>(&prepared, &[], pool, &mut memory)
            .unwrap();
        backend
            .download_buffer(pool, replay.buffer(), &mut observed[..N])
            .unwrap();
        verify(&observed, &want);
        drop(replay);
        let sdk_before = sdk.api();
        let mut native_owner = sdk.upload(&want);
        let mut raw_output = vec![0x5a; want.len() + 7];
        native_owner.read(&mut raw_output);
        assert_eq!(&raw_output[..want.len()], want);
        assert_eq!(&raw_output[want.len()..], &[0x5a; 7]);
        let mut short = vec![0x5a; want.len() - 1];
        assert!(!native_owner.try_read(&mut short));
        assert!(short.iter().all(|&byte| byte == 0x5a));
        native_owner.write(&mutate);
        let mut native_replay = sdk.upload(&want);
        native_replay.read(&mut raw_output);
        assert_eq!(&raw_output[..want.len()], want);
        global::clear_thread_cache().unwrap();
        backend
            .download_buffer(pool, escaped.buffer(), &mut observed[..N])
            .unwrap();
        verify(&observed, &[]);
        native_owner.read(&mut raw_output);
        assert_eq!(&raw_output[..want.len()], mutate);
        drop(native_replay);
        drop(native_owner);
        assert_eq!(sdk.api().delta(sdk_before).frees, 2);
        // Native escaped ownership retains its independently acquired context and library.
        let mut detached_native = {
            let temporary =
                Control::new(i32::try_from(backend.device_identity().device_id()).unwrap());
            temporary.upload(&want)
        };
        detached_native.read(&mut raw_output);
        assert_eq!(&raw_output[..want.len()], want);
        assert_eq!(&raw_output[want.len()..], &[0x5a; 7]);
        drop(detached_native);
        let before_unread_drop = sdk.api();
        drop(sdk.upload(&want));
        assert_eq!(sdk.api().delta(before_unread_drop).stream_waits, 1);
        for route in ["actual_source", "explicit_graph", "independent_hip"] {
            let profile = format!(
                "{mode:?}/{:?}/{:?}/{underflow:?}/{}/{N}/{kind}/{route}",
                options.compound_arithmetic,
                options.precision,
                T::LABEL
            );
            let mut warm = || {
                match route {
                    "actual_source" => source_owner::<T, N>(uniform)
                        .read_into(&mut observed)
                        .unwrap(),
                    "explicit_graph" => {
                        let owner = assessor
                            .execute_owned_program_output_from_inputs::<T, _>(
                                &prepared,
                                &[],
                                pool,
                                &mut memory,
                            )
                            .unwrap();
                        backend
                            .download_buffer(pool, owner.buffer(), &mut observed[..N])
                            .unwrap();
                    }
                    "independent_hip" => {
                        let mut owner = sdk.upload(&want);
                        owner.read(&mut raw_output);
                        assert_eq!(&raw_output[..want.len()], want);
                        assert_eq!(&raw_output[want.len()..], &[0x5a; 7]);
                    }
                    _ => unreachable!(),
                }
                if route != "independent_hip" {
                    verify(&observed, &want);
                }
            };
            warm();
            println!(
                "raw-integer-producer-semantic/{profile}: canonical bits/fresh escaped mutation/consume/replay/cache-clear/tails PASS"
            );
            #[cfg(feature = "allocation-census")]
            {
                let before = fusion_pcu_rocm::rocm_api_census();
                let sdk_before = sdk.api();
                let ((), heap) = super::allocations::measure(|| {
                    for _ in 0..64 {
                        warm();
                    }
                });
                let after = fusion_pcu_rocm::rocm_api_census();
                let native = sdk.api().delta(sdk_before);
                if route == "independent_hip" {
                    assert_eq!(
                        (
                            heap.alloc_calls,
                            heap.realloc_calls,
                            heap.dealloc_calls,
                            heap.requested_bytes
                        ),
                        (0, 0, 0, 0)
                    );
                    assert_eq!(
                        (
                            native.allocations,
                            native.frees,
                            native.uploads,
                            native.downloads,
                            native.selections
                        ),
                        (64, 64, 64, 64, 256)
                    );
                    assert_eq!(native.stream_waits, 128);
                    let replay_bytes = u64::try_from(want.len()).unwrap() * 64;
                    assert_eq!(native.uploaded_bytes, replay_bytes);
                    assert_eq!(native.downloaded_bytes, replay_bytes);
                }
                println!(
                    "raw-integer-producer-census/{profile}/64-immutable-replays: alloc={} realloc={} frees={} bytes={}; before={before:?}; after={after:?}; sdk={native:?}",
                    heap.alloc_calls, heap.realloc_calls, heap.dealloc_calls, heap.requested_bytes
                );
            }
            #[cfg(not(feature = "allocation-census"))]
            if !semantics {
                super::activity::activity_guard();
                let mut group = c.benchmark_group(format!("rocm_raw_integer_producers/{profile}"));
                group.sample_size(20);
                group.warm_up_time(Duration::from_millis(500));
                group.measurement_time(Duration::from_secs(2));
                group.bench_function("fresh_output_and_host_readback", |b| b.iter(&mut warm));
                group.finish();
            }
        }
        drop(prepared);
        backend
            .download_buffer(pool, escaped.buffer(), &mut observed[..N])
            .unwrap();
        verify(&observed, &[]);
        drop(escaped);
        println!(
            "raw-integer-producer-lifetime/{mode:?}/{:?}/{:?}/{underflow:?}/{}/{N}/{kind}: source cache-clear/graph plan-drop/native control-drop owners PASS",
            options.compound_arithmetic,
            options.precision,
            T::LABEL
        );
    }
}
fn cpu<T: RawSource, const N: usize>(
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
    underflow: PcuFloatUnderflowPolicy,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_mode: mode,
        numerical_options: options,
        float_underflow: underflow,
        ..Default::default()
    })
    .unwrap();
    T::capture::<N>(mode, options, underflow);
    for uniform in [false, true] {
        global::clear_thread_cache().unwrap();
        source_lifetime::<T, N>(uniform, &expected::<T, N>(uniform));
        println!(
            "raw-integer-producer-cpu-reference/{mode:?}/{:?}/{:?}/{underflow:?}/{}/{N}/{}: raw producer source and owner lifetime PASS",
            options.compound_arithmetic,
            options.precision,
            T::LABEL,
            if uniform { "uniform" } else { "constant" }
        );
    }
}
pub fn run(c: &mut Criterion) {
    let reference = std::env::var_os("PCU_RAW_INTEGER_PRODUCER_CPU_REFERENCE").is_some();
    let semantics = std::env::var_os("PCU_RAW_INTEGER_PRODUCER_SEMANTICS").is_some()
        || std::env::args().any(|arg| arg == "--test");
    if !semantics {
        super::activity::activity_guard();
    }
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
                for underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    macro_rules! widths {($($scalar:ty),+)=>{$(if let Some(backend) = &backend {
                        case::<$scalar,6>(c,backend,mode,options,underflow,semantics); case::<$scalar,65>(c,backend,mode,options,underflow,semantics); case::<$scalar,4096>(c,backend,mode,options,underflow,semantics);
                    } else { cpu::<$scalar,6>(mode,options,underflow); cpu::<$scalar,65>(mode,options,underflow); cpu::<$scalar,4096>(mode,options,underflow); })+};}
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
}

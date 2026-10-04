//! Genuine source/prepared/hand-IR routes against independently compiled native GLSL.
#[cfg(feature = "insights")]
#[path = "api_census/api_census.rs"]
mod api_census;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../composed_float_maps/ffi/ffi.rs"]
mod ffi;
#[path = "graph/graph.rs"]
mod graph;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{
    global, PcuBindingRef, PcuCheckedInteger, PcuCompoundArithmeticPolicy,
    PcuDispatchKernelIr, PcuFloatUnderflowPolicy, PcuHostArgument, PcuHostKernelBackend,
    PcuImplementationRequirements, PcuNumericalMode, PcuNumericalOptions,
    PcuPrecisionPolicy, PcuPreparedHostKernel, PcuRangePolicy, PcuReproducibility,
};
use fusion_pcu_vulkan::PcuVulkanBackend;
#[rustfmt::skip]
use std::{hint::black_box,sync::{Mutex,atomic::{AtomicUsize,Ordering}}};
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
static SCORES: AtomicUsize = AtomicUsize::new(0);
static EXPECTED: Mutex<Option<PcuImplementationRequirements>> = Mutex::new(None);
fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(
        Some(candidate.kernel.numerical_requirements),
        *EXPECTED.lock().unwrap()
    );
    fusion_pcu_spirv::validate_composed_integer_map(candidate.kernel)
        .or_else(|_| fusion_pcu_spirv::validate_one_effect_integer_map(candidate.kernel))
        .unwrap();
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
trait Integer: PcuCheckedInteger {
    fn small(value: u8) -> Self;
    fn edge(negative: bool) -> Self;
}
macro_rules! formats {
    ($($ty:ty),+) => {$(impl Integer for $ty {
        fn small(value: u8) -> Self {
            let mut bytes = [0; <$ty as pcu_facade::PcuScalar>::ENCODED_SIZE];
            bytes[0] = value;
            <Self as pcu_facade::PcuScalar>::decode_le(bytes)
        }
        fn edge(negative: bool) -> Self {
            let mut bytes = [255; <$ty as pcu_facade::PcuScalar>::ENCODED_SIZE];
            if matches!(<$ty as pcu_facade::PcuScalar>::TYPE,pcu_facade::PcuScalarType::I8 | pcu_facade::PcuScalarType::I16 | pcu_facade::PcuScalarType::I32 | pcu_facade::PcuScalarType::I64 | pcu_facade::PcuScalarType::I128 | pcu_facade::PcuScalarType::I256 | pcu_facade::PcuScalarType::I512) {
                if negative { bytes.fill(0); *bytes.last_mut().unwrap()=128; }
                else { *bytes.last_mut().unwrap()=127; }
            }
            <Self as pcu_facade::PcuScalar>::decode_le(bytes)
        }
    })+};
}
formats!(
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
    pcu_facade::PcuU256,
    pcu_facade::PcuI256,
    pcu_facade::PcuU512,
    pcu_facade::PcuI512
);
fn small<T: Integer>(value: u8) -> T {
    T::small(value)
}
fn values<T: Integer>(input: &mut [T], phase: usize) {
    for (lane, value) in input.iter_mut().enumerate() {
        *value = small::<T>(u8::try_from((phase + lane) % 4 + 1).unwrap());
    }
}
fn check<T: Integer, const ONE: bool>(input: &[T], output: &[T]) {
    // Allocation-free census assertion, including untouched tails.
    for (a, b) in input.iter().zip(output) {
        let want = if ONE {
            *a
        } else {
            a.pcu_checked_add(*a).unwrap().pcu_checked_mul(*a).unwrap()
        };
        assert_eq!(b.encode_le().as_ref(), want.encode_le().as_ref());
    }
    for value in &output[65..] {
        assert_eq!(
            value.encode_le().as_ref(),
            small::<T>(17).encode_le().as_ref()
        );
    }
}
fn compare<T: Integer, const ONE: bool>(
    criterion: &mut Criterion,
    name: &str,
    route: &str,
    mut call: impl FnMut(&[T], &mut [T]),
) {
    let mut input = [small::<T>(1); 65];
    let mut output = [small::<T>(17); 68];
    for phase in 0..3 {
        values(&mut input, phase);
        call(&input, &mut output);
        check::<T, ONE>(&input, &output);
    }
    if std::env::var_os("PCU_COMPOSED_CENSUS").is_some() {
        let scores = SCORES.load(Ordering::Relaxed);
        let counts = ffi::count_heap(|| {
            for phase in 0..64 {
                values(&mut input, phase);
                call(&input, &mut output);
                check::<T, ONE>(&input, &output);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(SCORES.load(Ordering::Relaxed), scores);
        println!("census {name}/{route}:64 changing calls0alloc0realloc0free0rescore");
    }
    #[cfg(feature = "insights")]
    api_census::warm(name, route, || {
        let counts = ffi::count_heap(|| {
            for phase in 0..64 {
                values(&mut input, phase);
                call(&input, &mut output);
                check::<T, ONE>(&input, &output);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
    });
    let mut group = criterion.benchmark_group(name);
    let mut phase = 0;
    group.bench_function(route, |bench| {
        bench.iter(|| {
            values(&mut input, phase);
            phase += 1;
            call(black_box(&input), black_box(&mut output));
            check::<T, ONE>(&input, &output);
        });
    });
    group.finish();
}
fn source_prepare<T: Integer, const ONE: bool>(
    backend: &PcuVulkanBackend,
    request: PcuImplementationRequirements,
) -> fusion_pcu_vulkan::PcuVulkanPreparedHost {
    let bindings = if ONE {
        source::dead_normal_bindings::<T>()
    } else {
        source::normal_bindings::<T>()
    };
    let prepare = |kernel: &PcuDispatchKernelIr<'_>| {
        assert_eq!(kernel.numerical_requirements, request);
        assert_eq!(
            fusion_pcu_spirv::validate_composed_integer_map(kernel)
                .or_else(|_| fusion_pcu_spirv::validate_one_effect_integer_map(kernel))
                .unwrap()
                .requirements(),
            request
        );
        backend.prepare_host_kernel(kernel).unwrap()
    };
    if ONE {
        if request.numerical_options.reproducibility == PcuReproducibility::PortableV1 {
            return source::__dead_portable_ir_with_float_underflow_policy::<T>(
                &bindings,
                request.float_underflow,
                request.range_policy,
                request,
            )
            .unwrap()
            .with_ir(prepare);
        }
        return source::__dead_normal_ir_with_float_underflow_policy::<T>(
            &bindings,
            request.float_underflow,
            request.range_policy,
            request,
        )
        .unwrap()
        .with_ir(prepare);
    }
    if request.numerical_options.reproducibility == PcuReproducibility::PortableV1 {
        source::__portable_ir_with_float_underflow_policy::<T>(
            &bindings,
            request.float_underflow,
            request.range_policy,
            request,
        )
        .unwrap()
        .with_ir(prepare)
    } else {
        source::__normal_ir_with_float_underflow_policy::<T>(
            &bindings,
            request.float_underflow,
            request.range_policy,
            request,
        )
        .unwrap()
        .with_ir(prepare)
    }
}
fn width<T: Integer, const ONE: bool>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
) {
    for uf in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let policy = match uf {
                PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
                PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
            };
            let mut native = if ONE {
                ffi::NativeComposed::one_effect
            } else {
                ffi::NativeComposed::new
            }(
                identity,
                65,
                policy,
                u32::from(range == PcuRangePolicy::Clamp),
                T::TYPE,
            )
            .unwrap();
            native.assert_policy(T::TYPE, uf, range);
            native_edges::<T, ONE>(&mut native, range);
            headers::<T, ONE>(criterion, backend, &mut native, uf, range);
        }
    }
}
fn native_edges<T: Integer, const ONE: bool>(
    native: &mut ffi::NativeComposed,
    range: PcuRangePolicy,
) {
    for negative in [false, true] {
        if negative
            && !matches!(
                T::TYPE,
                pcu_facade::PcuScalarType::I8
                    | pcu_facade::PcuScalarType::I16
                    | pcu_facade::PcuScalarType::I32
                    | pcu_facade::PcuScalarType::I64
                    | pcu_facade::PcuScalarType::I128
                    | pcu_facade::PcuScalarType::I256
                    | pcu_facade::PcuScalarType::I512
            )
        {
            continue;
        }
        let input = [T::edge(negative); 65];
        let mut output = [small::<T>(17); 68];
        let fault = native
            .call(ffi::bytes(&input), ffi::bytes_mut(&mut output))
            .unwrap()
            .unwrap();
        assert_eq!(
            fault,
            pcu_facade::PcuExecutionFault {
                invocation_id: 0,
                recovered: range == PcuRangePolicy::Clamp,
                kind: if negative {
                    pcu_facade::PcuExecutionFaultKind::ArithmeticUnderflow
                } else {
                    pcu_facade::PcuExecutionFaultKind::ArithmeticOverflow
                },
            }
        );
        let want = if range == PcuRangePolicy::Clamp {
            T::edge(ONE && negative)
        } else {
            small::<T>(17)
        };
        for value in &output[..65] {
            assert_eq!(value.encode_le().as_ref(), want.encode_le().as_ref());
        }
        let before = output;
        assert!(
            native
                .call(ffi::bytes(&input[..64]), ffi::bytes_mut(&mut output))
                .is_err()
        );
        for (a, b) in output.iter().zip(&before) {
            assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
        }
        assert_eq!(
            native
                .call(
                    ffi::bytes(&[small::<T>(2); 65]),
                    ffi::bytes_mut(&mut output)
                )
                .unwrap(),
            None
        );
        for value in &output[..65] {
            assert_eq!(
                value.encode_le().as_ref(),
                small::<T>(if ONE { 2 } else { 8 }).encode_le().as_ref()
            );
        }
        for value in &output[65..] {
            assert_eq!(
                value.encode_le().as_ref(),
                small::<T>(17).encode_le().as_ref()
            );
        }
        println!(
            "native_edge {:?}/{range:?}/{negative}/{ONE}:fault rollback Clamp tails retry",
            T::TYPE
        );
    }
}
fn headers<T: Integer, const ONE: bool>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    native: &mut ffi::NativeComposed,
    uf: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for reproducibility in [
                    PcuReproducibility::Unspecified,
                    PcuReproducibility::PortableV1,
                ] {
                    let request = PcuImplementationRequirements {
                        numerical_mode: mode,
                        float_underflow: uf,
                        range_policy: range,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            reproducibility,
                        },
                    };
                    *EXPECTED.lock().unwrap() = Some(request);
                    global::configure(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Vulkan,
                        numerical_mode: mode,
                        float_underflow: uf,
                        range_policy: range,
                        numerical_options: request.numerical_options,
                        score_invocation: Some(score),
                        ..Default::default()
                    })
                    .unwrap();
                    let name = format!(
                        "integer_composed/{ONE}/{:?}/{mode:?}/{compound:?}/{precision:?}/{reproducibility:?}/{uf:?}/{range:?}/65",
                        T::TYPE
                    );
                    let mut prepared = source_prepare::<T, ONE>(backend, request);
                    compare::<T, ONE>(criterion, &name, "prepared_source", |input, output| {
                        prepared
                            .call(&mut [
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 0), output),
                                PcuHostArgument::read(PcuBindingRef::new(0, 1), input),
                            ])
                            .unwrap();
                    });
                    compare::<T, ONE>(criterion, &name, "ordinary_source", |input, output| match (
                        ONE,
                        reproducibility,
                    ) {
                        (true, PcuReproducibility::PortableV1) => {
                            source::dead_portable::<T>(output, input).unwrap();
                        }
                        (true, _) => source::dead_normal::<T>(output, input).unwrap(),
                        (false, PcuReproducibility::PortableV1) => {
                            source::portable::<T>(output, input).unwrap();
                        }
                        (false, _) => source::normal::<T>(output, input).unwrap(),
                    });
                    let mut hand = graph::prepare::<T, 65, _, ONE>(backend, request);
                    compare::<T, ONE>(criterion, &name, "hand_ir", |input, output| {
                        hand.call(&mut [
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), output),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), input),
                        ])
                        .unwrap();
                    });
                    compare::<T, ONE>(criterion, &name, "native_glsl", |input, output| {
                        assert_eq!(
                            native
                                .call(ffi::bytes(input), ffi::bytes_mut(output))
                                .unwrap(),
                            None
                        );
                    });
                    global::clear_thread_cache().unwrap();
                }
            }
        }
    }
}
fn benchmark(criterion: &mut Criterion) {
    #[cfg(feature = "insights")]
    if std::env::var_os("PCU_COMPOSED_API_CENSUS").is_some() {
        api_census::run(|| benchmark_inner(criterion));
        return;
    }
    benchmark_inner(criterion);
}
fn benchmark_inner(criterion: &mut Criterion) {
    let (backend, identity) = device::selected();
    if std::env::var_os("PCU_WIDE_DISCARDED_ONLY").is_some() {
        width::<pcu_facade::PcuU256, true>(criterion, &backend, identity);
        width::<pcu_facade::PcuI256, true>(criterion, &backend, identity);
        width::<pcu_facade::PcuU512, true>(criterion, &backend, identity);
        width::<pcu_facade::PcuI512, true>(criterion, &backend, identity);
        return;
    }
    width::<u8, true>(criterion, &backend, identity);
    width::<i8, true>(criterion, &backend, identity);
    width::<u16, true>(criterion, &backend, identity);
    width::<i16, true>(criterion, &backend, identity);
    width::<u32, true>(criterion, &backend, identity);
    width::<i32, true>(criterion, &backend, identity);
    width::<u64, true>(criterion, &backend, identity);
    width::<i64, true>(criterion, &backend, identity);
    width::<u128, true>(criterion, &backend, identity);
    width::<i128, true>(criterion, &backend, identity);
    width::<pcu_facade::PcuU256, true>(criterion, &backend, identity);
    width::<pcu_facade::PcuI256, true>(criterion, &backend, identity);
    width::<pcu_facade::PcuU512, true>(criterion, &backend, identity);
    width::<pcu_facade::PcuI512, true>(criterion, &backend, identity);
    width::<u8, false>(criterion, &backend, identity);
    width::<i8, false>(criterion, &backend, identity);
    width::<u16, false>(criterion, &backend, identity);
    width::<i16, false>(criterion, &backend, identity);
    width::<u32, false>(criterion, &backend, identity);
    width::<i32, false>(criterion, &backend, identity);
    width::<u64, false>(criterion, &backend, identity);
    width::<i64, false>(criterion, &backend, identity);
    width::<u128, false>(criterion, &backend, identity);
    width::<i128, false>(criterion, &backend, identity);
    width::<pcu_facade::PcuU256, false>(criterion, &backend, identity);
    width::<pcu_facade::PcuI256, false>(criterion, &backend, identity);
    width::<pcu_facade::PcuU512, false>(criterion, &backend, identity);
    width::<pcu_facade::PcuI512, false>(criterion, &backend, identity);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);

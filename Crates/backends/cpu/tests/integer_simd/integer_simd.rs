//! Public cold ISA identity and whole-call publication, including source capture and retry.
#[path = "../../benches/integer_simd/native/native.rs"]
mod native;
#[path = "../../benches/integer_simd/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuProcessor,PcuCpuImplementation,PcuCpuCheckedInteger,PcuCpuCheckedIntegerError};
#[rustfmt::skip]
use pcu_facade::{global,PcuHostKernelBackend,PcuPreparedHostKernel,PcuHostArgument,PcuBindingRef,PcuDispatchIntegerBinaryOp,PcuRangePolicy};
fn instruction() -> PcuCpuImplementation {
    let p = PcuCpuProcessor::detect();
    if p.features().avx512f && p.features().avx512bw {
        PcuCpuImplementation::Avx512
    } else if p.features().avx2 {
        PcuCpuImplementation::Avx2
    } else if p.features().sse2 {
        PcuCpuImplementation::Sse2
    } else {
        assert!(p.features().neon);
        PcuCpuImplementation::Neon
    }
}
fn width<T: native::Native>() {
    for isa in [
        PcuCpuImplementation::Sse2,
        PcuCpuImplementation::Avx2,
        PcuCpuImplementation::Avx512,
        PcuCpuImplementation::Neon,
    ] {
        if PcuCpuProcessor::detect().require(isa).is_err()
            || (isa == PcuCpuImplementation::Avx512
                && !PcuCpuProcessor::detect().features().avx512bw)
        {
            continue;
        }
        let backend =
            PcuCpuCheckedInteger::<T>::with_implementation(PcuCpuProcessor::detect(), isa).unwrap();
        macro_rules! operation {
            ($entry:ident,$ir:ident,$bindings:ident,$op:ident,$range:ident) => {{
                let bindings = source::$bindings::<T>();
                let builder = source::$ir::<T, 65>(&bindings).unwrap();
                let mut plan = backend.prepare_host_kernel(&builder.ir()).unwrap();
                assert_eq!(plan.implementation(), isa);
                assert!((1024..=1151).contains(&plan.local_id()));
                assert_eq!(plan.implementation_revision(), 1);
                let mut left = [T::small(10); 65];
                let right = [T::small(2); 65];
                let sentinel = T::small(77);
                for index in [0, 1, 7, 8, 15, 16, 31, 32, 63, 64] {
                    left.fill(T::small(10));
                    left[index] =
                        if PcuDispatchIntegerBinaryOp::$op == PcuDispatchIntegerBinaryOp::Add {
                            T::maximum()
                        } else {
                            T::minimum()
                        };
                    let mut expected = [sentinel; 68];
                    let notice = native::execute::<T, 65>(
                        &left,
                        &right,
                        &mut expected,
                        PcuDispatchIntegerBinaryOp::$op,
                        PcuRangePolicy::$range,
                    );
                    let mut output = [sentinel; 68];
                    let result = plan
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
                        ])
                        .map_err(|error| match error {
                            PcuCpuCheckedIntegerError::Fault(f) => f,
                            other => panic!("unexpected typed SIMD{other:?}"),
                        });
                    assert_eq!(result, notice);
                    assert_eq!(output, expected);
                    output.fill(sentinel);
                    assert_eq!(
                        source::$entry::<T, 65>(&left, &right, &mut output)
                            .map_err(|error| error.arithmetic_fault().unwrap()),
                        notice
                    );
                    assert_eq!(output, expected);
                    let saved = output;
                    assert!(
                        plan.call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &left[..64]),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output)
                        ])
                        .is_err()
                    );
                    assert_eq!(output, saved);
                    left.fill(T::small(10));
                    source::$entry::<T, 65>(&left, &right, &mut output).unwrap();
                    assert_eq!(output[65..], [sentinel; 3]);
                }
            }};
        }
        operation!(add, add_ir, add_bindings, Add, Reject);
        operation!(sub, sub_ir, sub_bindings, Sub, Reject);
        operation!(add_clamp, add_clamp_ir, add_clamp_bindings, Add, Clamp);
        operation!(sub_clamp, sub_clamp_ir, sub_clamp_bindings, Sub, Clamp);
        let left = [T::small(10); 65];
        let right = T::small(2);
        let mut output = [T::small(77); 68];
        source::broadcast::<T, 65>(&left, &right, &mut output).unwrap();
        assert_eq!(output[..65], [T::small(12); 65]);
        assert_eq!(output[65..], [T::small(77); 3]);
    }
}
#[test]
fn native_eight_public_source_transactions() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    width::<i8>();
    width::<u8>();
    width::<i16>();
    width::<u16>();
    width::<i32>();
    width::<u32>();
    width::<i64>();
    width::<u64>();
}
#[test]
fn explicit_requests_never_substitute() {
    let p = PcuCpuProcessor::scalar();
    assert!(matches!(
        PcuCpuCheckedInteger::<u32>::with_implementation(p, PcuCpuImplementation::Sse2),
        Err(PcuCpuCheckedIntegerError::ImplementationUnavailable(_))
    ));
    let p = PcuCpuProcessor::detect();
    assert!(matches!(
        PcuCpuCheckedInteger::<u32>::with_implementation(p, PcuCpuImplementation::Avx),
        Err(PcuCpuCheckedIntegerError::UnsupportedProfile)
    ));
    for isa in [PcuCpuImplementation::Avx2, PcuCpuImplementation::Avx512] {
        if p.require(isa).is_ok() && (isa != PcuCpuImplementation::Avx512 || p.features().avx512bw)
        {
            assert!(PcuCpuCheckedInteger::<u32>::with_implementation(p, isa).is_ok());
        } else {
            assert!(PcuCpuCheckedInteger::<u32>::with_implementation(p, isa).is_err());
        }
    }
    let native = PcuCpuCheckedInteger::<u32>::with_implementation(p, instruction()).unwrap();
    let bindings = source::mul_bindings::<u32>();
    let builder = source::mul_ir::<u32, 65>(&bindings).unwrap();
    assert!(matches!(
        native.prepare_host_kernel(&builder.ir()),
        Err(PcuCpuCheckedIntegerError::UnsupportedProfile)
    ));
    let backend = PcuCpuCheckedInteger::<u128>::with_implementation(p, instruction()).unwrap();
    let bindings = source::add_bindings::<u128>();
    let builder = source::add_ir::<u128, 65>(&bindings).unwrap();
    assert!(matches!(
        backend.prepare_host_kernel(&builder.ir()),
        Err(PcuCpuCheckedIntegerError::UnsupportedProfile)
    ));
}

#[rustfmt::skip]
use pcu_facade::{PcuDeviceIdentity,PcuObjectRef,PcuObjectKind,PcuProviderId,PcuExecutorId,PcuImplementationRequest,PcuImplementationOffers,PcuCostBoundary,PcuNumericalMode,PcuReproducibility,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy};
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuHostOffers,PcuCpuHostBackend};
fn cold_width<T: native::Native>() {
    let processor = PcuCpuProcessor::detect();
    let typed = PcuCpuCheckedInteger::<T>::with_implementation(processor, instruction()).unwrap();
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(9),
        generation: 1,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let hosted = PcuCpuHostOffers::new(
        PcuCpuHostBackend::new(processor, PcuCpuImplementation::Scalar).unwrap(),
        device,
        PcuExecutorId(0),
    );
    macro_rules! exact {
        ($bindings:ident,$ir:ident) => {{
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, 17>(&bindings).unwrap();
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for compound in [
                    PcuCompoundArithmeticPolicy::Checked,
                    PcuCompoundArithmeticPolicy::BackendDefined,
                ] {
                    for precision in [
                        PcuPrecisionPolicy::Preserve,
                        PcuPrecisionPolicy::BackendOptimized,
                    ] {
                        let mut kernel = builder.ir();
                        kernel.numerical_requirements.numerical_mode = mode;
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .compound_arithmetic = compound;
                        kernel.numerical_requirements.numerical_options.precision = precision;
                        let plan = typed.prepare_host_kernel(&kernel).unwrap();
                        let mut request = PcuImplementationRequest {
                            device,
                            executor: PcuExecutorId(0),
                            boundary: PcuCostBoundary::Host,
                            operation: &kernel,
                            requirements: kernel.numerical_requirements,
                        };
                        let mut output = [None];
                        assert_eq!(hosted.implementation_offers(&request, &mut []).unwrap(), 1);
                        assert_eq!(
                            hosted.implementation_offers(&request, &mut output).unwrap(),
                            1
                        );
                        let offer = output[0].unwrap();
                        assert_eq!(offer.implementation.local_id, plan.local_id());
                        assert_eq!(offer.implementation.revision, 1);
                        assert_eq!(offer.workspace_bytes, Some(0));
                        assert_eq!(offer.requirements, request.requirements);
                        request.requirements.numerical_mode = if mode == PcuNumericalMode::Strict {
                            PcuNumericalMode::Boundary
                        } else {
                            PcuNumericalMode::Strict
                        };
                        output = [None];
                        assert_eq!(
                            hosted.implementation_offers(&request, &mut output).unwrap(),
                            0
                        );
                        assert_eq!(output, [None]);
                        request.requirements = kernel.numerical_requirements;
                        request.boundary = PcuCostBoundary::Resident;
                        assert_eq!(
                            hosted.implementation_offers(&request, &mut output).unwrap(),
                            0
                        );
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                        let portable = typed.prepare_host_kernel(&kernel).unwrap();
                        let ids = match portable.implementation() {
                            PcuCpuImplementation::Sse2 => 4224..=4255,
                            PcuCpuImplementation::Neon => 4352..=4383,
                            PcuCpuImplementation::Avx2 => 4480..=4511,
                            PcuCpuImplementation::Avx512 => 4608..=4639,
                            _ => panic!("expected the selected SIMD implementation"),
                        };
                        assert!(ids.contains(&portable.local_id()));
                        assert_eq!(portable.implementation(), plan.implementation());
                    }
                }
            }
        }};
    }
    exact!(add_bindings, add_ir);
    exact!(sub_bindings, sub_ir);
    exact!(add_clamp_bindings, add_clamp_ir);
    exact!(sub_clamp_bindings, sub_clamp_ir);
}
#[test]
fn cold_exact_offers_keep_independent_permissions() {
    cold_width::<i8>();
    cold_width::<u8>();
    cold_width::<i16>();
    cold_width::<u16>();
    cold_width::<i32>();
    cold_width::<u32>();
    cold_width::<i64>();
    cold_width::<u64>();
}

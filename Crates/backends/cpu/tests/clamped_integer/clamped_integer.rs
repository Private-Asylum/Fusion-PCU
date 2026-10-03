//! Independent full-width oracle for complete observable integer Clamp publication.
#[path = "../wide_integer/oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
mod source;
use oracle::Wide;
use fusion_pcu_cpu::{PcuCpuCheckedInteger, PcuCpuCheckedIntegerError, PcuCpuHostBackend};
#[rustfmt::skip]
use pcu_facade::{global,PcuI256,PcuU256,PcuI512,PcuU512,PcuDispatchIntegerBinaryOp,PcuExecutionFault,PcuExecutionFaultKind,PcuHostKernelBackend,PcuPreparedHostKernel,PcuBindingRef,PcuHostArgument};
fn expected<T: Wide>(
    left: T,
    right: T,
    op: PcuDispatchIntegerBinaryOp,
) -> (T, Option<PcuExecutionFaultKind>) {
    match oracle::evaluate(left, right, op) {
        Ok(v) => (v, None),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow) => (
            oracle::minimum(),
            Some(PcuExecutionFaultKind::ArithmeticUnderflow),
        ),
        Err(PcuExecutionFaultKind::ArithmeticOverflow) => (
            oracle::maximum(),
            Some(PcuExecutionFaultKind::ArithmeticOverflow),
        ),
        Err(other) => panic!("integer range-only oracle returned {other:?}"),
    }
}
fn samples<T: Wide, const N: usize>() -> (Vec<T>, Vec<T>) {
    let zero = oracle::small(0);
    let mut left = vec![zero; N];
    let mut right = vec![zero; N];
    let edges = [
        zero,
        oracle::small(1),
        oracle::small(2),
        oracle::minimum(),
        oracle::maximum(),
    ];
    let mut state = 0xc01a_5734_eca9_8342u64;
    for lane in 0..N {
        if T::BYTES == 1 {
            let mut a = [0; 64];
            let mut b = [0; 64];
            a[0] = u8::try_from(lane / 256).unwrap();
            b[0] = u8::try_from(lane % 256).unwrap();
            left[lane] = T::from_bytes(a);
            right[lane] = T::from_bytes(b);
        } else if lane < 25 {
            left[lane] = edges[lane / 5];
            right[lane] = edges[lane % 5];
        } else {
            for destination in [&mut left[lane], &mut right[lane]] {
                let mut bytes = [0; 64];
                for byte in &mut bytes[..T::BYTES] {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *byte = state.to_le_bytes()[0];
                }
                *destination = T::from_bytes(bytes);
            }
        }
    }
    (left, right)
}
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)] // Complete per-width source/graph publication follows one independent dataset and notice ordering.
fn compare<T: Wide, const N: usize>() -> usize {
    let (left, right) = samples::<T, N>();
    let sentinel = oracle::small(17);
    let mut output = vec![sentinel; N + 3];
    let backend = PcuCpuCheckedInteger::<T>::new();
    macro_rules! operation {
        ($entry:ident,$prepare:ident,$ir:ident,$bindings:ident,$op:ident) => {{
            let mut prepared = source::$prepare::<T, N, _>(&backend).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
            let mut expected_output = Vec::with_capacity(N);
            let mut notice = None;
            for lane in 0..N {
                let (value, kind) =
                    expected(left[lane], right[lane], PcuDispatchIntegerBinaryOp::$op);
                expected_output.push(value);
                if let Some(kind) = kind {
                    notice.get_or_insert(PcuExecutionFault {
                        kind,
                        invocation_id: u64::try_from(lane).unwrap(),
                        recovered: true,
                    });
                }
            }
            let expected_result =
                notice.map_or(Ok(()), |fault| Err(PcuCpuCheckedIntegerError::Fault(fault)));
            assert_eq!(prepared(&left, &right, &mut output), expected_result);
            assert_eq!(output[..N], expected_output);
            assert_eq!(output[N..], [sentinel; 3]);
            assert_eq!(
                graph.call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output)
                ]),
                expected_result
            );
            assert_eq!(output[..N], expected_output);
            let before = output.clone();
            assert!(prepared(&left[..N - 1], &right, &mut output).is_err());
            assert_eq!(output, before);
            assert_eq!(prepared(&left, &right, &mut output), expected_result);
        }};
    }
    operation!(add, add_prepare, add_ir, add_bindings, Add);
    operation!(sub, sub_prepare, sub_ir, sub_bindings, Sub);
    operation!(mul, mul_prepare, mul_ir, mul_bindings, Mul);
    N * 3
}
#[test]
fn exhaustive_eight_bit_and_full_bit_fourteen_width_independent_oracle() {
    let count = compare::<i8, 65536>()
        + compare::<u8, 65536>()
        + compare::<i16, 4096>()
        + compare::<u16, 4096>()
        + compare::<i32, 4096>()
        + compare::<u32, 4096>()
        + compare::<i64, 4096>()
        + compare::<u64, 4096>()
        + compare::<i128, 4096>()
        + compare::<u128, 4096>()
        + compare::<PcuI256, 4096>()
        + compare::<PcuU256, 4096>()
        + compare::<PcuI512, 4096>()
        + compare::<PcuU512, 4096>();
    assert_eq!(count, 540_672);
}
fn source_transaction<T: Wide>() {
    let backend = PcuCpuHostBackend::scalar();
    let max = oracle::maximum::<T>();
    let one = oracle::small(1);
    let two = oracle::small(2);
    let sentinel = oracle::small(17);
    let mut prepared = source::grid_prepare::<T, 7, _>(&backend).unwrap();
    let mut output = [sentinel; 10];
    let mut left = [one; 7];
    left[2] = max;
    left[5] = max;
    let error = prepared(&left, &two, &mut output)
        .unwrap_err()
        .fault()
        .unwrap();
    assert_eq!(
        error,
        PcuExecutionFault {
            recovered: true,
            invocation_id: 2,
            kind: PcuExecutionFaultKind::ArithmeticOverflow
        }
    );
    assert_eq!(output[..7], [two, two, max, two, two, max, two]);
    assert_eq!(output[7..], [sentinel; 3]);
    let before = output;
    assert!(prepared(&left[..6], &two, &mut output).is_err());
    assert_eq!(output, before);
    left.fill(one);
    prepared(&left, &two, &mut output).unwrap();
    assert_eq!(output[..7], [two; 7]);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    source::add::<T, 7>(&[max; 7], &[one; 7], &mut output).unwrap_err();
    assert_eq!(output[..7], [max; 7]);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
#[test]
fn fourteen_width_strict_grid_broadcast_preflight_retry_and_ordinary_source() {
    source_transaction::<i8>();
    source_transaction::<u8>();
    source_transaction::<i16>();
    source_transaction::<u16>();
    source_transaction::<i32>();
    source_transaction::<u32>();
    source_transaction::<i64>();
    source_transaction::<u64>();
    source_transaction::<i128>();
    source_transaction::<u128>();
    source_transaction::<PcuI256>();
    source_transaction::<PcuU256>();
    source_transaction::<PcuI512>();
    source_transaction::<PcuU512>();
}

#[rustfmt::skip]
use pcu_facade::{PcuDeviceIdentity,PcuObjectRef,PcuObjectKind,PcuProviderId,PcuExecutorId,PcuImplementationRequest,PcuImplementationOffers,PcuCostBoundary,PcuRangePolicy,PcuNumericalMode,PcuReproducibility,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy};
use fusion_pcu_cpu::{PcuCpuIntegerOffers, PcuCpuHostOffers};
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)] // Exact typed/host tuples, IDs and negative requests stay in one cold matrix audit.
fn offers<T: Wide>(base: u32) {
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(9),
        generation: 1,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let typed = PcuCpuIntegerOffers::<T>::new(device, PcuExecutorId(0));
    let hosted = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), device, PcuExecutorId(0));
    macro_rules! operation {
        ($ir:ident,$bindings:ident,$offset:literal) => {{
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, 7>(&bindings).unwrap();
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
                        let mut request = PcuImplementationRequest {
                            device,
                            executor: PcuExecutorId(0),
                            boundary: PcuCostBoundary::Host,
                            operation: &kernel,
                            requirements: kernel.numerical_requirements,
                        };
                        let mut output = [None];
                        assert_eq!(typed.implementation_offers(&request, &mut []).unwrap(), 1);
                        assert_eq!(
                            typed.implementation_offers(&request, &mut output).unwrap(),
                            1
                        );
                        let offer = output[0].unwrap();
                        assert_eq!(offer.implementation.local_id, base + $offset);
                        assert_eq!(offer.implementation.revision, 1);
                        assert_eq!(offer.workspace_bytes, Some(0));
                        assert_eq!(offer.requirements, request.requirements);
                        let mut host_output = [None];
                        assert_eq!(
                            hosted
                                .implementation_offers(&request, &mut host_output)
                                .unwrap(),
                            1
                        );
                        assert_eq!(host_output[0].unwrap().implementation, offer.implementation);
                        request.requirements.range_policy = PcuRangePolicy::Reject;
                        output = [None];
                        assert_eq!(
                            typed.implementation_offers(&request, &mut output).unwrap(),
                            0
                        );
                        assert_eq!(output, [None]);
                        request.requirements = kernel.numerical_requirements;
                        request.boundary = PcuCostBoundary::Resident;
                        assert_eq!(
                            typed.implementation_offers(&request, &mut output).unwrap(),
                            0
                        );
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                        let portable = PcuCpuCheckedInteger::<T>::new()
                            .prepare_host_kernel(&kernel)
                            .unwrap();
                        assert!((4096..=4179).contains(&portable.local_id()));
                        assert_eq!(portable.implementation_revision(), 1);
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::Unspecified;
                        kernel.numerical_requirements.range_policy = PcuRangePolicy::Reject;
                        assert!(
                            PcuCpuCheckedInteger::<T>::new()
                                .prepare_host_kernel(&kernel)
                                .is_err()
                        );
                    }
                }
            }
        }};
    }
    operation!(add_ir, add_bindings, 0);
    operation!(sub_ir, sub_bindings, 1);
    operation!(mul_ir, mul_bindings, 2);
}
#[test]
fn disjoint_clamp_ids_revision_exact_tuple_and_portable() {
    offers::<i8>(320);
    offers::<u8>(323);
    offers::<i16>(326);
    offers::<u16>(329);
    offers::<i32>(332);
    offers::<u32>(335);
    offers::<i64>(338);
    offers::<u64>(341);
    offers::<i128>(344);
    offers::<u128>(347);
    offers::<PcuI256>(350);
    offers::<PcuU256>(353);
    offers::<PcuI512>(356);
    offers::<PcuU512>(359);
}

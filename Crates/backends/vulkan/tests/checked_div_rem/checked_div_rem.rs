//! Actual U32 quotient/remainder against wider native-integer arithmetic and whole-call publication.
extern crate pcu_facade as fusion_pcu;
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../../benches/checked_div_rem/ffi/ffi.rs"]
#[allow(dead_code)]
mod ffi;
#[path = "../../../spirv/tests/checked_div_rem/graph/graph.rs"]
mod graph;
#[path = "../../../cpu/tests/checked_div_rem/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use pcu_facade::{global,PcuExecutionFaultKind as Kind,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel};
use fusion_pcu_vulkan::{PcuVulkanBackend, PcuVulkanError, PcuVulkanPreparedHost};
#[path = "../../benches/checked_div_rem/oracle/oracle.rs"]
mod oracle;
use oracle::{Integer, expected};
fn call<T: Integer>(
    plan: &mut PcuVulkanPreparedHost,
    a: &[T],
    b: &[T],
    q: &mut [T],
    r: &mut [T],
) -> Result<(), PcuVulkanError> {
    plan.call(&mut [
        PcuHostArgument::read_write(graph::OUTPUTS[1], r),
        PcuHostArgument::read(graph::INPUTS[0], a),
        PcuHostArgument::read_write(graph::OUTPUTS[0], q),
        PcuHostArgument::read(graph::INPUTS[1], b),
    ])
}
#[allow(clippy::too_many_lines, clippy::many_single_char_names)] // One complete arithmetic/transaction fixture; a,b,q,r are conventional division operands.
fn proof<T: Integer>(backend: &PcuVulkanBackend, identity: pcu_facade::PcuStableDeviceIdentity) {
    let count = if T::HOST_SIZE == 1 { 65536 } else { 16384 };
    let mut a = Vec::with_capacity(count);
    let mut b = Vec::with_capacity(count);
    let mut random = 0x9856_abcd_1234_5678_u64;
    for i in 0..count {
        let (x, y) = if T::HOST_SIZE == 1 {
            (
                u64::try_from(i / 256).unwrap(),
                u64::try_from(i % 256).unwrap(),
            )
        } else {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let x = random;
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            (x, random)
        };
        a.push(T::from_raw(x));
        b.push(T::from_raw(y));
    }
    let sentinel = T::from_raw(27);
    let mut q = vec![sentinel; count + 3];
    let mut r = q.clone();
    let mut codes = vec![0; count];
    let mut native = ffi::NativeDivRem::new(
        identity,
        u32::try_from(count).unwrap(),
        T::SIGNED,
        T::HOST_SIZE,
    )
    .unwrap();
    native
        .diagnostics(
            ffi::bytes(&a),
            ffi::bytes(&b),
            ffi::bytes_mut(&mut q),
            ffi::bytes_mut(&mut r),
            &mut codes,
        )
        .unwrap();
    for i in 0..count {
        let (wq, wr, status) = expected(a[i], b[i]);
        assert_eq!(codes[i], status);
        if status == 0 {
            assert_eq!((q[i], r[i]), (wq, wr));
        }
    }
    let mut g = graph::Graph::new(T::TYPE, u32::try_from(count).unwrap());
    let mut plan = g.with(|k| backend.prepare_host_kernel(k).unwrap());
    q.fill(sentinel);
    r.fill(sentinel);
    let result = call(&mut plan, &a, &b, &mut q, &mut r);
    if let Some(i) = codes.iter().position(|v| *v != 0) {
        let PcuVulkanError::Fault(fault) = result.unwrap_err() else {
            panic!("expected arithmetic fault")
        };
        assert_eq!(fault.invocation_id, u64::try_from(i).unwrap());
        assert_eq!(
            fault.kind,
            if codes[i] == 4 {
                Kind::DivideByZero
            } else {
                Kind::SignedDivisionOverflow
            }
        );
        assert_eq!(q, vec![sentinel; count + 3]);
        assert_eq!(r, vec![sentinel; count + 3]);
    } else {
        result.unwrap();
    }
    let mut valid_a = Vec::new();
    let mut valid_b = Vec::new();
    for (&a, &b) in a.iter().zip(&b) {
        if expected(a, b).2 == 0 {
            valid_a.push(a);
            valid_b.push(b);
        }
    }
    g.extent = u32::try_from(valid_a.len()).unwrap();
    g.grid = true;
    plan = g.with(|k| backend.prepare_host_kernel(k).unwrap());
    call(&mut plan, &valid_a, &valid_b, &mut q, &mut r).unwrap();
    for i in 0..valid_a.len() {
        let (wq, wr, _) = expected(valid_a[i], valid_b[i]);
        assert_eq!((q[i], r[i]), (wq, wr));
    }
    assert!(
        q[valid_a.len()..]
            .iter()
            .chain(&r[valid_a.len()..])
            .all(|v| *v == sentinel)
    );
    println!("{:?} independent raw pairs={count}", T::TYPE);
}
#[test]
#[ignore = "requires actual Vulkan GPU and validation layer"]
fn complete_eight_bit_and_wider_independent_division() {
    let (backend, identity) = device::selected();
    proof::<i8>(&backend, identity);
    proof::<u8>(&backend, identity);
    proof::<i16>(&backend, identity);
    proof::<u16>(&backend, identity);
    proof::<i32>(&backend, identity);
    proof::<u32>(&backend, identity);
    proof::<i64>(&backend, identity);
    proof::<u64>(&backend, identity);
}
#[test]
#[ignore = "requires actual Vulkan GPU"]
fn genuine_eight_width_source_prepared_grid_transaction() {
    let (backend, _) = device::selected();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    macro_rules! width {
        ($t:ty,$module:ident) => {{
            let a = [7 as $t; 65];
            let mut b = [3 as $t; 65];
            let mut q = [27 as $t; 68];
            let mut r = q;
            let mut prepared = source::$module::direct_prepare::<65, _>(&backend).unwrap();
            prepared(&a, &b, &mut q, &mut r).unwrap();
            assert_eq!(q[..65], [2 as $t; 65]);
            assert_eq!(r[..65], [1 as $t; 65]);
            source::$module::direct::<65>(&a, &b, &mut q, &mut r).unwrap();
            source::$module::grid::<65>(&a, &b, &mut q, &mut r).unwrap();
            source::$module::strict::<65>(&a, &b, &mut q, &mut r).unwrap();
            let saved = (q, r);
            b[2] = 0;
            let e = source::$module::direct::<65>(&a, &b, &mut q, &mut r)
                .unwrap_err()
                .arithmetic_fault()
                .unwrap();
            assert_eq!(
                (e.invocation_id, e.kind, e.recovered),
                (2, Kind::DivideByZero, false)
            );
            assert_eq!((q, r), saved);
            assert!(prepared(&a[..64], &b, &mut q, &mut r).is_err());
            assert!(prepared(&a, &b, &mut q, &mut r[..64]).is_err());
            assert_eq!((q, r), saved);
            b[2] = 3;
            prepared(&a, &b, &mut q, &mut r).unwrap();
            assert_eq!(q[65..], [27 as $t; 3]);
            assert_eq!(r[65..], [27 as $t; 3]);
        }};
    }
    width!(i8, i8_source);
    width!(u8, u8_source);
    width!(i16, i16_source);
    width!(u16, u16_source);
    width!(i32, i32_source);
    width!(u32, u32_source);
    width!(i64, i64_source);
    width!(u64, u64_source);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[test]
#[ignore = "requires actual Vulkan compute device"]
#[allow(clippy::too_many_lines)] // The full cold tuple matrix and independent SSA routes share one admission proof.
fn exact_cold_offers_and_actual_operand_routing() {
    #[rustfmt::skip]
    use pcu_facade::{PcuImplementationOffers,PcuImplementationRequest,PcuExecutorId,PcuCostBoundary,PcuNumericalMode,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuReproducibility,PcuRangePolicy,PcuFloatUnderflowPolicy,PcuScalarType};
    let (backend, _) = device::selected();
    let mut profiles = 0;
    for (ordinal, scalar) in [
        PcuScalarType::I8,
        PcuScalarType::U8,
        PcuScalarType::I16,
        PcuScalarType::U16,
        PcuScalarType::I32,
        PcuScalarType::U32,
        PcuScalarType::I64,
        PcuScalarType::U64,
    ]
    .into_iter()
    .enumerate()
    {
        for grid in [false, true] {
            let mut g = graph::Graph::new(scalar, 65);
            g.grid = grid;
            g.with(|kernel| {
                for underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                        for compound in [
                            PcuCompoundArithmeticPolicy::Checked,
                            PcuCompoundArithmeticPolicy::BackendDefined,
                        ] {
                            for precision in [
                                PcuPrecisionPolicy::Preserve,
                                PcuPrecisionPolicy::BackendOptimized,
                            ] {
                                let mut kernel = *kernel;
                                kernel.numerical_requirements.float_underflow = underflow;
                                kernel.numerical_requirements.numerical_mode = mode;
                                kernel
                                    .numerical_requirements
                                    .numerical_options
                                    .compound_arithmetic = compound;
                                kernel.numerical_requirements.numerical_options.precision =
                                    precision;
                                let request = PcuImplementationRequest {
                                    device: backend.device_identity().unwrap(),
                                    executor: PcuExecutorId(0),
                                    boundary: PcuCostBoundary::Host,
                                    operation: &kernel,
                                    requirements: kernel.numerical_requirements,
                                };
                                assert_eq!(
                                    backend.implementation_offers(&request, &mut []).unwrap(),
                                    1
                                );
                                let mut offers = [None];
                                backend
                                    .implementation_offers(&request, &mut offers)
                                    .unwrap();
                                let offer = offers[0].unwrap();
                                assert_eq!(
                                    offer.implementation.local_id,
                                    608 + u32::try_from(ordinal).unwrap()
                                );
                                assert_eq!(offer.implementation.revision, 1);
                                offer.validate_request(&request).unwrap();
                                profiles += 1;
                                for portable in [false, true] {
                                    let mut bad = kernel;
                                    if portable {
                                        bad.numerical_requirements
                                            .numerical_options
                                            .reproducibility = PcuReproducibility::PortableV1;
                                    } else {
                                        bad.numerical_requirements.range_policy =
                                            PcuRangePolicy::Clamp;
                                    }
                                    let unsupported = PcuImplementationRequest {
                                        operation: &bad,
                                        requirements: bad.numerical_requirements,
                                        ..request
                                    };
                                    let count = backend
                                        .implementation_offers(&unsupported, &mut [])
                                        .unwrap();
                                    if portable {
                                        assert_eq!(count, 1);
                                        let mut offers = [None];
                                        backend
                                            .implementation_offers(&unsupported, &mut offers)
                                            .unwrap();
                                        let offer = offers[0].unwrap();
                                        assert_eq!(
                                            offer.implementation.local_id,
                                            12544 + u32::try_from(ordinal).unwrap()
                                        );
                                        assert_eq!(offer.implementation.revision, 1);
                                        offer.validate_request(&unsupported).unwrap();
                                        let mismatch = PcuImplementationRequest {
                                            requirements: request.requirements,
                                            ..unsupported
                                        };
                                        assert_eq!(
                                            backend
                                                .implementation_offers(&mismatch, &mut [])
                                                .unwrap(),
                                            0
                                        );
                                        assert!(backend.prepare_host_kernel(&bad).is_ok());
                                    } else {
                                        assert_eq!(count, 0);
                                        assert!(backend.prepare_host_kernel(&bad).is_err());
                                    }
                                }
                                let mismatch = PcuImplementationRequest {
                                    requirements: pcu_facade::PcuImplementationRequirements {
                                        range_policy: PcuRangePolicy::Clamp,
                                        ..request.requirements
                                    },
                                    ..request
                                };
                                assert_eq!(
                                    backend.implementation_offers(&mismatch, &mut []).unwrap(),
                                    0
                                );
                            }
                        }
                    }
                }
            });
        }
    }
    assert_eq!(profiles, 384);
    for grid in [false, true] {
        for routing in 0..3 {
            let mut graph = graph::Graph::new(PcuScalarType::I64, 65);
            graph.grid = grid;
            graph.routing = routing;
            let mut plan = graph.with(|k| backend.prepare_host_kernel(k).unwrap());
            let left = [-17_i64; 65];
            let right = [5_i64; 65];
            let mut quotient = [27; 68];
            let mut remainder = [29; 68];
            call(&mut plan, &left, &right, &mut quotient, &mut remainder).unwrap();
            let (want_q, want_r) = match routing {
                0 => (-3, -2),
                1 => (0, 5),
                _ => (1, 0),
            };
            assert_eq!(quotient[..65], [want_q; 65]);
            assert_eq!(remainder[..65], [want_r; 65]);
            assert_eq!(quotient[65..], [27; 3]);
            assert_eq!(remainder[65..], [29; 3]);
        }
    }
}

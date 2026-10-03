//! Actual fourteen-width integer maps against independent base-256 arithmetic and native compilation.
extern crate pcu_facade as fusion_pcu;
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../../benches/checked_integer/ffi/ffi.rs"]
#[allow(dead_code)]
mod ffi;
#[path = "../../../rocm/benches/wide_integer/oracle/oracle.rs"]
#[allow(dead_code)]
// Goldens are read independently; benchmark-specific input builders are unused here.
mod goldens;
#[path = "../../../spirv/tests/checked_integer/graph/graph.rs"]
mod graph;
#[path = "../../../cpu/tests/wide_integer/oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
#[allow(dead_code)]
mod source;
use oracle::Wide;
#[rustfmt::skip]
use pcu_facade::{global,PcuI256,PcuU256,PcuI512,PcuU512,PcuScalar,PcuDispatchIntegerBinaryOp,PcuRangePolicy,PcuExecutionFault,PcuExecutionFaultKind,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel};
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanError,PcuVulkanPreparedHost};
const OPS: [PcuDispatchIntegerBinaryOp; 3] = [
    PcuDispatchIntegerBinaryOp::Add,
    PcuDispatchIntegerBinaryOp::Sub,
    PcuDispatchIntegerBinaryOp::Mul,
];
macro_rules! widths {($f:ident,$($a:expr),*)=>{{
    $f::<i8>($($a),*);$f::<u8>($($a),*);$f::<i16>($($a),*);$f::<u16>($($a),*);$f::<i32>($($a),*);$f::<u32>($($a),*);$f::<i64>($($a),*);$f::<u64>($($a),*);$f::<i128>($($a),*);$f::<u128>($($a),*);$f::<PcuI256>($($a),*);$f::<PcuU256>($($a),*);$f::<PcuI512>($($a),*);$f::<PcuU512>($($a),*);
}};}
fn fault(error: PcuVulkanError) -> PcuExecutionFault {
    match error {
        PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected native error {other:?}"),
    }
}
fn expected<T: Wide>(left: T, right: T, op: PcuDispatchIntegerBinaryOp) -> (T, u32) {
    match oracle::evaluate(left, right, op) {
        Ok(value) => (value, 0),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow) => (oracle::minimum(), 2),
        Err(PcuExecutionFaultKind::ArithmeticOverflow) => (oracle::maximum(), 3),
        Err(other) => panic!("range-only oracle returned {other:?}"),
    }
}
fn samples<T: Wide>(count: usize, offset: usize) -> (Vec<T>, Vec<T>) {
    let zero = oracle::small(0);
    let edges = [
        zero,
        oracle::small(1),
        oracle::small(2),
        oracle::minimum(),
        oracle::maximum(),
        T::from_bytes([255; 64]),
    ];
    let mut state = 0xc01a_5734_eca9_8342_u64;
    let mut values = (vec![zero; count], vec![zero; count]);
    for lane in 0..count {
        if T::BYTES == 1 {
            let mut a = [0; 64];
            let mut b = [0; 64];
            a[0] = u8::try_from((lane + offset) / 256).unwrap();
            b[0] = u8::try_from((lane + offset) % 256).unwrap();
            values.0[lane] = T::from_bytes(a);
            values.1[lane] = T::from_bytes(b);
        } else if lane < 36 {
            values.0[lane] = edges[lane / 6];
            values.1[lane] = edges[lane % 6];
        } else {
            for destination in [&mut values.0[lane], &mut values.1[lane]] {
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
    values
}
fn call<T: PcuScalar>(
    prepared: &mut PcuVulkanPreparedHost,
    left: &[T],
    right: &[T],
    output: &mut [T],
) -> Result<(), PcuVulkanError> {
    prepared.call(&mut [
        PcuHostArgument::read_write(graph::OUTPUT, output),
        PcuHostArgument::read(graph::INPUTS[1], right),
        PcuHostArgument::read(graph::INPUTS[0], left),
    ])
}
#[allow(clippy::too_many_lines)] // Each independent dataset checks private native statuses and public all-or-nothing publication together.
fn independent<T: Wide>(backend: &PcuVulkanBackend, identity: pcu_facade::PcuStableDeviceIdentity) {
    const N: usize = 4096;
    let sentinel = oracle::small(17);
    let mut output = vec![sentinel; N + 3];
    let mut native_output = output.clone();
    let mut statuses = vec![0; N];
    for (operation, op) in OPS.into_iter().enumerate() {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let g = graph::Graph::new(T::TYPE, u32::try_from(N).unwrap(), op, range);
            let mut prepared = g.with(|k| backend.prepare_host_kernel(k).unwrap());
            let PcuVulkanPreparedHost::Integer(map) = &prepared else {
                panic!("exact integer owner required")
            };
            assert_eq!(map.profile().inputs, graph::INPUTS);
            assert_eq!(map.memory_realizations().unwrap().len(), 4);
            let mut native = ffi::NativeInteger::new(
                identity,
                u32::try_from(N).unwrap(),
                u32::try_from(operation).unwrap(),
                range == PcuRangePolicy::Clamp,
                T::SIGNED,
                T::BYTES,
            )
            .unwrap();
            let total = if T::BYTES == 1 { 65536 } else { N };
            for offset in (0..total).step_by(N) {
                let (left, right) = samples::<T>(N, offset);
                native
                    .diagnostics(
                        ffi::bytes(&left),
                        ffi::bytes(&right),
                        ffi::bytes_mut(&mut native_output),
                        &mut statuses,
                    )
                    .unwrap();
                let mut want = Vec::with_capacity(N);
                let mut notice = None;
                for lane in 0..N {
                    let (value, status) = expected(left[lane], right[lane], op);
                    want.push(value);
                    assert_eq!(
                        statuses[lane],
                        status,
                        "{:?} {op:?}/{range:?} lane{}",
                        T::TYPE,
                        lane + offset
                    );
                    if status == 0 || range == PcuRangePolicy::Clamp {
                        assert_eq!(
                            native_output[lane],
                            value,
                            "{:?} {op:?}/{range:?} lane{}",
                            T::TYPE,
                            lane + offset
                        );
                    }
                    if status != 0 {
                        notice.get_or_insert_with(|| PcuExecutionFault {
                            kind: if status == 2 {
                                PcuExecutionFaultKind::ArithmeticUnderflow
                            } else {
                                PcuExecutionFaultKind::ArithmeticOverflow
                            },
                            invocation_id: u64::try_from(lane).unwrap(),
                            recovered: range == PcuRangePolicy::Clamp,
                        });
                    }
                }
                output.fill(sentinel);
                let result = call(&mut prepared, &left, &right, &mut output).map_err(fault);
                assert_eq!(result, notice.map_or(Ok(()), Err));
                if notice.is_none() || range == PcuRangePolicy::Clamp {
                    assert_eq!(output[..N], want);
                } else {
                    assert_eq!(output[..N], [sentinel; N]);
                }
                assert_eq!(output[N..], [sentinel; 3]);
                assert_eq!(native_output[N..], [sentinel; 3]);
            }
            // A changed, valid batch proves reuse after a range fault and no stale status/payload.
            let left = [oracle::small::<T>(2); N];
            let right = [oracle::small::<T>(1); N];
            call(&mut prepared, &left, &right, &mut output).unwrap();
            assert_eq!(output[..N], [expected(left[0], right[0], op).0; N]);
        }
    }
    println!(
        "independent {:?}: {} base256 pairs x3ops x2range; native statuses/results and provider publication",
        T::TYPE,
        if T::BYTES == 1 { 65536 } else { N }
    );
}
#[test]
#[ignore = "requires actual Vulkan compute GPU and GLSL compiler"]
fn complete_eight_bit_and_full_width_independent_arithmetic() {
    let (backend, identity) = device::selected();
    widths!(independent, &backend, identity);
}
#[allow(clippy::too_many_lines)] // The source/prepared/ordinary transaction matrix shares each concrete checked operation's generated schema.
fn source_profiles<T: Wide>(backend: &PcuVulkanBackend) {
    const N: usize = 65;
    let sentinel = oracle::small(17);
    let left = [oracle::small::<T>(2); N];
    let right = [oracle::small::<T>(1); N];
    let mut output = [sentinel; N + 3];
    macro_rules! operation {
        ($module:ident,$entry:ident,$prepare:ident,$op:ident,$clamp:expr) => {{
            let mut prepared = source::$module::$prepare::<T, N, _>(backend).unwrap();
            prepared(&left, &right, &mut output).unwrap();
            assert_eq!(
                output[..N],
                [expected(left[0], right[0], PcuDispatchIntegerBinaryOp::$op).0; N]
            );
            source::$module::$entry::<T, N>(&left, &right, &mut output).unwrap();
            assert_eq!(output[N..], [sentinel; 3]);
            let before = output;
            assert!(prepared(&left[..N - 1], &right, &mut output).is_err());
            assert_eq!(output, before);
            assert!(prepared(&left, &right, &mut output[..N - 1]).is_err());
            assert_eq!(output, before);
            let mut bad = left;
            bad[2] = if PcuDispatchIntegerBinaryOp::$op == PcuDispatchIntegerBinaryOp::Sub {
                oracle::minimum()
            } else {
                oracle::maximum()
            };
            let mut rhs = right;
            rhs[2] = oracle::small(2);
            let result = prepared(&bad, &rhs, &mut output)
                .map_err(fault)
                .unwrap_err();
            assert_eq!(result.invocation_id, 2);
            assert_eq!(result.recovered, $clamp);
            if $clamp {
                for lane in 0..N {
                    assert_eq!(
                        output[lane],
                        expected(bad[lane], rhs[lane], PcuDispatchIntegerBinaryOp::$op).0
                    );
                }
            } else {
                assert_eq!(output, before);
            }
            prepared(&left, &right, &mut output).unwrap();
            assert_eq!(output, before);
        }};
    }
    operation!(reject, add, add_prepare, Add, false);
    operation!(reject, sub, sub_prepare, Sub, false);
    operation!(reject, mul, mul_prepare, Mul, false);
    operation!(clamp, add, add_prepare, Add, true);
    operation!(clamp, sub, sub_prepare, Sub, true);
    operation!(clamp, mul, mul_prepare, Mul, true);
    let mut grid = source::reject::grid_prepare::<T, N, _>(backend).unwrap();
    grid(&left, &right, &mut output).unwrap();
    assert_eq!(output[..N], [oracle::small::<T>(3); N]);
    let scalar = oracle::small::<T>(2);
    let mut scale = source::reject::scale_prepare::<T, N, _>(backend).unwrap();
    scale(&left, &scalar, &mut output).unwrap();
    assert_eq!(output[..N], [oracle::small::<T>(4); N]);
    let mut clamp_grid = source::clamp::grid_prepare::<T, N, _>(backend).unwrap();
    clamp_grid(&left, &scalar, &mut output).unwrap();
    assert_eq!(output[..N], [oracle::small::<T>(4); N]);
    source::reject::grid::<T, N>(&left, &right, &mut output).unwrap();
    source::reject::scale::<T, N>(&left, &scalar, &mut output).unwrap();
    source::clamp::grid::<T, N>(&left, &scalar, &mut output).unwrap();
    // Noncanonical binding and swapped SSA must keep the original readonly span requirements.
    let mut g = graph::Graph::new(
        T::TYPE,
        u32::try_from(N).unwrap(),
        PcuDispatchIntegerBinaryOp::Sub,
        PcuRangePolicy::Reject,
    );
    g.grid = true;
    g.broadcast = true;
    g.swapped = true;
    let mut swapped = g.with(|k| backend.prepare_host_kernel(k).unwrap());
    call(&mut swapped, &right, &[scalar], &mut output).unwrap();
    assert_eq!(output[..N], [oracle::small::<T>(1); N]);
    let before = output;
    assert!(call(&mut swapped, &right, &[], &mut output).is_err());
    assert_eq!(output, before);
    assert_eq!(output[N..], [sentinel; 3]);
}
#[test]
#[ignore = "requires actual Vulkan compute GPU"]
fn ordinary_prepared_grid_broadcast_preflight_fault_tail_and_retry() {
    let (backend, _) = device::selected();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    widths!(source_profiles, &backend);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

fn golden_width<T: Wide + goldens::Format>(
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
) {
    let sentinel = oracle::small::<T>(17);
    let mut output = [sentinel; 3];
    let mut native_output = [sentinel; 3];
    let mut status = [0];
    for (operation, op) in OPS.into_iter().enumerate() {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let mut prepared = graph::Graph::new(T::TYPE, 1, op, range)
                .with(|k| backend.prepare_host_kernel(k).unwrap());
            let mut native = ffi::NativeInteger::new(
                identity,
                1,
                u32::try_from(operation).unwrap(),
                range == PcuRangePolicy::Clamp,
                T::SIGNED,
                T::BYTES,
            )
            .unwrap();
            for (left, right, golden, golden_code) in
                goldens::rows::<T>(u32::try_from(operation).unwrap())
            {
                // The independent HIP/CUDA goldens encode range underflow as 4;
                // this Vulkan status protocol encodes the same fault as 2.
                let code = match golden_code {
                    0 | 3 => golden_code,
                    4 => 2,
                    other => panic!("unrecognized independent golden status{other}"),
                };
                output.fill(sentinel);
                native
                    .diagnostics(
                        ffi::bytes(&[left]),
                        ffi::bytes(&[right]),
                        ffi::bytes_mut(&mut native_output),
                        &mut status,
                    )
                    .unwrap();
                assert_eq!(status, [code]);
                let want = match code {
                    0 => golden,
                    2 => oracle::minimum(),
                    3 => oracle::maximum(),
                    other => panic!("unrecognized independent golden status{other}"),
                };
                let result = call(&mut prepared, &[left], &[right], &mut output).map_err(fault);
                if code == 0 {
                    result.unwrap();
                    assert_eq!(output[0], golden);
                } else {
                    let notice = result.unwrap_err();
                    assert_eq!(notice.invocation_id, 0);
                    assert_eq!(notice.recovered, range == PcuRangePolicy::Clamp);
                    assert_eq!(
                        notice.kind,
                        if code == 2 {
                            PcuExecutionFaultKind::ArithmeticUnderflow
                        } else {
                            PcuExecutionFaultKind::ArithmeticOverflow
                        }
                    );
                    assert_eq!(
                        output[0],
                        if range == PcuRangePolicy::Clamp {
                            want
                        } else {
                            sentinel
                        }
                    );
                }
                if code == 0 || range == PcuRangePolicy::Clamp {
                    assert_eq!(native_output[0], want);
                }
                assert_eq!(output[1..], [sentinel; 2]);
                assert_eq!(native_output[1..], [sentinel; 2]);
            }
        }
    }
}
#[test]
#[ignore = "requires actual Vulkan compute GPU and GLSL compiler"]
fn wide_arbitrary_precision_golden_bytes_and_range_boundaries() {
    let (backend, identity) = device::selected();
    golden_width::<i128>(&backend, identity);
    golden_width::<u128>(&backend, identity);
    golden_width::<PcuI256>(&backend, identity);
    golden_width::<PcuU256>(&backend, identity);
    golden_width::<PcuI512>(&backend, identity);
    golden_width::<PcuU512>(&backend, identity);
}

#[test]
#[ignore = "requires explicitly available Vulkan compute device"]
#[allow(clippy::too_many_lines)] // Exact full numerical tuples and unsupported mutations share the same frozen profile.
fn exact_fourteen_format_cold_offers_and_closed_portable() {
    use pcu_facade::{
        PcuImplementationOffers, PcuImplementationRequest, PcuExecutorId, PcuCostBoundary,
        PcuNumericalMode, PcuCompoundArithmeticPolicy, PcuPrecisionPolicy, PcuReproducibility,
    };
    let (backend, _) = device::selected();
    let mut profiles = 0;
    for (format, scalar) in [
        pcu_facade::PcuScalarType::I8,
        pcu_facade::PcuScalarType::U8,
        pcu_facade::PcuScalarType::I16,
        pcu_facade::PcuScalarType::U16,
        pcu_facade::PcuScalarType::I32,
        pcu_facade::PcuScalarType::U32,
        pcu_facade::PcuScalarType::I64,
        pcu_facade::PcuScalarType::U64,
        pcu_facade::PcuScalarType::I128,
        pcu_facade::PcuScalarType::U128,
        pcu_facade::PcuScalarType::I256,
        pcu_facade::PcuScalarType::U256,
        pcu_facade::PcuScalarType::I512,
        pcu_facade::PcuScalarType::U512,
    ]
    .into_iter()
    .enumerate()
    {
        for (op_code, op) in OPS.into_iter().enumerate() {
            for policy in [
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                pcu_facade::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                pcu_facade::PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                for (range_code, range) in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp]
                    .into_iter()
                    .enumerate()
                {
                    let mut g = graph::Graph::new(scalar, 65, op, range);
                    g.grid = range_code == 1;
                    g.broadcast = op_code == 1;
                    g.with(|kernel| {
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
                                    kernel.numerical_requirements.float_underflow = policy;
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
                                        512 + u32::try_from(format * 6 + range_code * 3 + op_code)
                                            .unwrap()
                                    );
                                    assert_eq!(offer.implementation.revision, 1);
                                    offer.validate_request(&request).unwrap();
                                    profiles += 1;
                                    let mut bad = kernel;
                                    bad.numerical_requirements.numerical_options.reproducibility =
                                        PcuReproducibility::PortableV1;
                                    let unsupported = PcuImplementationRequest {
                                        operation: &bad,
                                        requirements: bad.numerical_requirements,
                                        ..request
                                    };
                                    assert_eq!(
                                        backend
                                            .implementation_offers(&unsupported, &mut [])
                                            .unwrap(),
                                        1
                                    );
                                    assert!(backend.prepare_host_kernel(&bad).is_ok());
                                    let mismatch = PcuImplementationRequest {
                                        requirements: pcu_facade::PcuImplementationRequirements {
                                            range_policy: if range == PcuRangePolicy::Clamp {
                                                PcuRangePolicy::Reject
                                            } else {
                                                PcuRangePolicy::Clamp
                                            },
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
                    });
                }
            }
        }
    }
    assert_eq!(profiles, 2016);
}

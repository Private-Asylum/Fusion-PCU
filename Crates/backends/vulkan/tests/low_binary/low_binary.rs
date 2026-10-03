//! Full encoding oracle and observable public publication for packed low-format arithmetic.
#[path = "../../benches/low_binary/ffi/ffi.rs"]
#[allow(dead_code)]
// Test binary uses diagnostics; paired benchmark uses allocation and stage controls.
mod ffi;
#[path = "../../../spirv/tests/checked_binary/support/support.rs"]
mod graph;
#[path = "../../../cpu/tests/low_precision/oracle/oracle.rs"]
#[allow(dead_code)] // Shared oracle also exposes Reject evaluation to CPU peers.
mod oracle;
#[path = "../../../cpu/tests/low_precision/source/source.rs"]
#[allow(dead_code)] // Other source profiles are exercised by backend parity and paired benchmarks.
mod reject_source;
#[path = "../clamped_binary/source/source.rs"]
#[allow(dead_code)] // Canonical generic family is shared with paired controls.
mod source;
use oracle::Low;
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanDiscovery,PcuVulkanError};
#[rustfmt::skip]
use pcu_facade::{global,PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuDispatchFloatBinaryOp,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuScalarType,PcuExecutionFault,PcuExecutionFaultKind,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuDeviceClass,PcuDeviceDescriptor,PcuObjectKind,PcuObjectRef,PcuProviderDescriptor,PcuProviderId,PcuProviderReadiness,PcuProviderStatus,PcuRuntimeDiscovery,PcuStableDeviceIdentity,PcuTargetDescriptor};
const OPS: [PcuDispatchFloatBinaryOp; 4] = [
    PcuDispatchFloatBinaryOp::Add,
    PcuDispatchFloatBinaryOp::Sub,
    PcuDispatchFloatBinaryOp::Mul,
    PcuDispatchFloatBinaryOp::Div,
];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
];
fn selected() -> (PcuVulkanBackend, PcuStableDeviceIdentity) {
    let discovery = PcuVulkanDiscovery::discover().unwrap();
    let empty = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Target,
        id: 0,
    };
    let readiness = PcuProviderReadiness {
        status: PcuProviderStatus::Unavailable,
        reason: None,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness,
    }];
    discovery.providers(&mut providers).unwrap();
    let mut targets = [PcuTargetDescriptor {
        reference: empty,
        name: "",
        readiness,
    }];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .unwrap();
    let mut devices = [PcuDeviceDescriptor {
        reference: empty,
        target: empty,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }];
    assert!(
        discovery
            .devices(targets[0].reference, &mut devices)
            .unwrap()
            > 0
    );
    let selected = devices[0].reference;
    (
        PcuVulkanBackend::open(&discovery, selected).unwrap(),
        discovery
            .device_facts(selected)
            .unwrap()
            .stable_identity
            .unwrap(),
    )
}

fn format<T: Low>() -> u32 {
    match T::TYPE {
        PcuScalarType::F16 => 0,
        PcuScalarType::BF16 => 1,
        PcuScalarType::F8E4M3FN => 2,
        PcuScalarType::F8E5M2 => 3,
        _ => unreachable!("closed low test family"),
    }
}
fn status(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::InvalidFloatingOperand => 1,
        PcuExecutionFaultKind::ArithmeticUnderflow => 2,
        PcuExecutionFaultKind::ArithmeticOverflow => 3,
        PcuExecutionFaultKind::DivideByZero => 4,
        PcuExecutionFaultKind::SignedDivisionOverflow => unreachable!("closed float fault set"),
    }
}
#[allow(clippy::too_many_lines)] // Complete encoding/product profile comparison retains one native owner per operation and policy.
fn encoding<T: Low>(identity: &PcuStableDeviceIdentity) -> usize {
    let f = T::FORMAT;
    let mut left = Vec::new();
    let mut right = Vec::new();
    if f.sign == 0x80 {
        for a in 0..256 {
            for b in 0..256 {
                left.push(T::from_bits(a));
                right.push(T::from_bits(b));
            }
        }
    } else {
        let edges = [
            0,
            f.sign,
            1,
            f.sign | 1,
            (1 << f.fraction) - 1,
            1 << f.fraction,
            f.max,
            f.max - 1,
            f.sign | f.max,
            f.max + 1,
            f.sign - 1,
        ];
        for a in 0..=u16::MAX {
            for b in edges
                .into_iter()
                .chain(core::iter::once(a.wrapping_mul(40503).wrapping_add(17)))
            {
                left.push(T::from_bits(a));
                right.push(T::from_bits(b));
            }
        }
    }
    let mut output = vec![T::from_bits(0); left.len()];
    let mut statuses = vec![0; left.len()];
    let mut comparisons = 0;
    for (operation, op) in OPS.into_iter().enumerate() {
        for (policy_id, policy) in POLICIES.into_iter().enumerate() {
            let mut native = ffi::NativeBinary::new(
                *identity,
                u32::try_from(left.len()).unwrap(),
                u32::try_from(operation).unwrap(),
                u32::try_from(policy_id).unwrap(),
                format::<T>(),
            )
            .unwrap();
            native
                .diagnostics(
                    ffi::bytes(&left),
                    ffi::bytes(&right),
                    ffi::bytes_mut(&mut output),
                    &mut statuses,
                )
                .unwrap();
            for lane in 0..left.len() {
                let expected =
                    f.evaluate_clamped(left[lane].bits(), right[lane].bits(), op, policy);
                let (bits, code) = match expected {
                    Ok((bits, None)) => (bits, 0),
                    Ok((bits, Some(kind))) => (bits, status(kind)),
                    Err(kind) => (0, status(kind)),
                };
                assert_eq!(
                    (output[lane].bits(), statuses[lane]),
                    (bits, code),
                    "{:?} {op:?} {policy:?} left={:#x} right={:#x}",
                    T::TYPE,
                    left[lane].bits(),
                    right[lane].bits()
                );
                comparisons += 1;
            }
        }
    }
    comparisons
}
#[test]
#[ignore = "requires a physical Vulkan compute device and GLSL compiler"]
fn complete_four_format_private_result_and_status_oracle() {
    let (_, identity) = selected();
    let count = encoding::<PcuF16Bits>(&identity)
        + encoding::<PcuBf16Bits>(&identity)
        + encoding::<PcuF8E4M3FnBits>(&identity)
        + encoding::<PcuF8E5M2Bits>(&identity);
    assert_eq!(count, 20_447_232);
}
fn call<T: Low>(
    prepared: &mut impl PcuPreparedHostKernel<Error = PcuVulkanError>,
    left: &[T],
    right: &[T],
    output: &mut [T],
) -> Result<(), PcuVulkanError> {
    prepared.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), left),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), right),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
    ])
}
fn fault(error: &PcuVulkanError) -> PcuExecutionFault {
    if let PcuVulkanError::Fault(fault) = error {
        *fault
    } else {
        panic!("unexpected {error:?}")
    }
}
#[allow(clippy::too_many_lines)] // Fatal precedence, publication, tails, preflight and retry form one owner lifecycle.
fn publication<T: Low>(backend: &PcuVulkanBackend) {
    let f = T::FORMAT;
    let one = T::from_bits(u16::try_from(f.bias).unwrap() << f.fraction);
    let two = T::from_bits(u16::try_from(f.bias + 1).unwrap() << f.fraction);
    let max = T::from_bits(f.max);
    let sentinel = T::from_bits(17);
    for policy in POLICIES {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let mut g = graph::Graph::new(7, PcuDispatchFloatBinaryOp::Mul, policy);
            g.scalar = T::TYPE;
            g.range = range;
            let mut prepared = g
                .with(|kernel| backend.prepare_host_kernel(kernel))
                .unwrap();
            let mut left = [max; 7];
            let mut right = [two; 7];
            let mut output = [sentinel; 10];
            let error = fault(&call(&mut prepared, &left, &right, &mut output).unwrap_err());
            assert_eq!(error.kind, PcuExecutionFaultKind::ArithmeticOverflow);
            assert_eq!(error.recovered, range == PcuRangePolicy::Clamp);
            assert_eq!(error.invocation_id, 0);
            assert_eq!(
                output[..7],
                [if range == PcuRangePolicy::Clamp {
                    max
                } else {
                    sentinel
                }; 7]
            );
            assert_eq!(&output[7..], &[sentinel; 3]);
            for bad_lane in [0, 6] {
                left[bad_lane] = T::from_bits(f.max + 1);
                let before = output;
                let error = fault(&call(&mut prepared, &left, &right, &mut output).unwrap_err());
                assert_eq!(
                    error.kind,
                    if range == PcuRangePolicy::Reject && bad_lane != 0 {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    } else {
                        PcuExecutionFaultKind::InvalidFloatingOperand
                    }
                );
                assert!(!error.recovered);
                assert_eq!(
                    error.invocation_id,
                    if range == PcuRangePolicy::Reject {
                        0
                    } else {
                        u64::try_from(bad_lane).unwrap()
                    }
                );
                assert_eq!(output, before);
                left[bad_lane] = max;
            }
            let before = output;
            assert!(call(&mut prepared, &left[..6], &right, &mut output).is_err());
            assert_eq!(output, before);
            left.fill(one);
            right.fill(two);
            call(&mut prepared, &left, &right, &mut output).unwrap();
            assert_eq!(&output[..7], &[two; 7]);
            assert_eq!(&output[7..], &[sentinel; 3]);
        }
    }
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn four_format_public_transactions_odd_lengths_and_retry() {
    let (backend, _) = selected();
    publication::<PcuF16Bits>(&backend);
    publication::<PcuBf16Bits>(&backend);
    publication::<PcuF8E4M3FnBits>(&backend);
    publication::<PcuF8E5M2Bits>(&backend);
}
fn source_paths<T: Low>(backend: &PcuVulkanBackend) {
    let f = T::FORMAT;
    let one = T::from_bits(u16::try_from(f.bias).unwrap() << f.fraction);
    let two = T::from_bits(u16::try_from(f.bias + 1).unwrap() << f.fraction);
    let left = [two; 7];
    let right = [one; 7];
    let mut output = [T::from_bits(17); 10];
    let mut prepared = reject_source::div_prepare::<T, 7, _>(backend).unwrap();
    prepared(&left, &right, &mut output).unwrap();
    reject_source::grid::<T, 7>(&left, &right, &mut output).unwrap();
    source::scale::<T, 7>(&left, &one, &mut output).unwrap();
    assert_eq!(&output[..7], &left);
    assert_eq!(&output[7..], &[T::from_bits(17); 3]);
    let max = T::from_bits(f.max);
    let e = source::mul::<T, 7>(&[max; 7], &[two; 7], &mut output).unwrap_err();
    assert!(
        matches!(e,global::PcuExecutionError::ArithmeticFault(fault) if fault.recovered && fault.kind==PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(&output[..7], &[max; 7]);
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn genuine_low_source_prepared_grid_broadcast_and_clamp() {
    let (backend, _) = selected();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    source_paths::<PcuF16Bits>(&backend);
    source_paths::<PcuBf16Bits>(&backend);
    source_paths::<PcuF8E4M3FnBits>(&backend);
    source_paths::<PcuF8E5M2Bits>(&backend);
}

#[allow(clippy::too_many_lines)] // All original input layouts and SSA routes qualify the same packed ownership seam.
fn layouts<T: Low>(backend: &PcuVulkanBackend) {
    let f = T::FORMAT;
    let one = u16::try_from(f.bias).unwrap() << f.fraction;
    let left =
        core::array::from_fn::<_, 7, _>(|i| T::from_bits(one + u16::try_from(i + 1).unwrap()));
    let right =
        core::array::from_fn::<_, 7, _>(|i| T::from_bits(one + u16::try_from(i + 8).unwrap()));
    for op in OPS {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            for grid in [false, true] {
                for broadcast in [[false, false], [true, false], [false, true], [true, true]] {
                    for operands in [[1, 2], [2, 1], [1, 1], [2, 2]] {
                        let mut g =
                            graph::Graph::new(7, op, PcuFloatUnderflowPolicy::IeeeAfterRounding);
                        g.scalar = T::TYPE;
                        g.range = range;
                        g.grid = grid;
                        g.reverse_loads = grid;
                        g.broadcast = broadcast;
                        g.operands = operands;
                        let mut prepared = g
                            .with(|kernel| backend.prepare_host_kernel(kernel))
                            .unwrap();
                        let left = &left[..if broadcast[0] { 1 } else { 7 }];
                        let right = &right[..if broadcast[1] { 1 } else { 7 }];
                        let mut output = [T::from_bits(17); 9];
                        call(&mut prepared, left, right, &mut output).unwrap();
                        for (lane, value) in output[..7].iter().enumerate() {
                            let operand = |binding: u16| {
                                let bank = usize::from(binding - 1);
                                let index = if broadcast[bank] { 0 } else { lane };
                                if bank == 0 { left[index] } else { right[index] }
                            };
                            assert_eq!(
                                value.bits(),
                                f.evaluate(
                                    operand(operands[0]).bits(),
                                    operand(operands[1]).bits(),
                                    op,
                                    PcuFloatUnderflowPolicy::IeeeAfterRounding
                                )
                                .unwrap()
                            );
                        }
                        assert_eq!(&output[7..], &[T::from_bits(17); 2]);
                        let before = output;
                        assert!(
                            call(&mut prepared, &left[..left.len() - 1], right, &mut output)
                                .is_err()
                        );
                        assert_eq!(output, before);
                        call(&mut prepared, left, right, &mut output).unwrap();
                        assert_eq!(output, before);
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn packed_original_binding_grid_broadcast_repeated_and_swapped_ssa() {
    let (backend, _) = selected();
    layouts::<PcuF16Bits>(&backend);
    layouts::<PcuBf16Bits>(&backend);
    layouts::<PcuF8E4M3FnBits>(&backend);
    layouts::<PcuF8E5M2Bits>(&backend);
}

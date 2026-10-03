//! Actual annotated identity source against the explicit Metal host authoring seam.
use pcu_facade::pcu;
use fusion_pcu_metal::MetalSession;

// These source cases exercise process-wide policy inheritance. Keep their configuration and
// calls together so another test cannot install a different numerical contract mid-case.
static SOURCE_POLICY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn source_policy_guard() -> std::sync::MutexGuard<'static, ()> {
    SOURCE_POLICY_LOCK.lock().unwrap()
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn identity<const N: usize>(input: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[test]
#[ignore = "Requires actual macOS Metal device and runs annotated source on GPU."]
fn annotated_source_current_inputs_and_tail_preservation() {
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut source = identity_prepare::<3, _>(&session).unwrap();
    let mut output = [91_u32; 5];
    source(&[1, 2, u32::MAX], &mut output).unwrap();
    assert_eq!(output, [1, 2, u32::MAX, 91, 91]);
    source(&[7, 8, 9], &mut output).unwrap();
    assert_eq!(output, [7, 8, 9, 91, 91]);
    assert!(matches!(
        source(&[1], &mut output),
        Err(pcu_facade::PcuHostDispatchError::BufferTooSmall(_))
    ));
    assert_eq!(output, [7, 8, 9, 91, 91]);
    drop(session);
    source(&[2, 3, 4], &mut output).unwrap();
    assert_eq!(output, [2, 3, 4, 91, 91]);
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = N, flag(reject_subnormal_result), crate_path = ::pcu_facade)]
fn negate_tight<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[test]
#[ignore = "Requires actual macOS Metal device and runs annotated checked Neg source on GPU."]
fn annotated_neg_preserves_bits_fault_output_and_retry() {
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut source = negate_prepare::<3, _>(&session).unwrap();
    let mut output = [42.0_f32; 5];
    source(&[0.0, -0.0, f32::from_bits(1)], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [
            0x8000_0000,
            0,
            0x8000_0001,
            42.0_f32.to_bits(),
            42.0_f32.to_bits()
        ]
    );
    let before = output.map(f32::to_bits);
    assert!(
        matches!(source(&[1.0, f32::from_bits(0x7f80_0001), f32::INFINITY], &mut output), Err(pcu_facade::PcuHostDispatchError::Backend(fusion_pcu_metal::MetalError::Arithmetic(fault))) if fault.kind == pcu_facade::PcuExecutionFaultKind::InvalidFloatingOperand && fault.invocation_id == 1)
    );
    assert_eq!(output.map(f32::to_bits), before);
    source(&[1.0, -2.0, 3.0], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [-1.0_f32, 2.0, -3.0, 42.0, 42.0].map(f32::to_bits)
    );
    let mut tight = negate_tight_prepare::<3, _>(&session).unwrap();
    assert!(
        matches!(tight(&[0.0, f32::from_bits(1), f32::INFINITY], &mut output), Err(pcu_facade::PcuHostDispatchError::Backend(fusion_pcu_metal::MetalError::Arithmetic(fault))) if fault.kind == pcu_facade::PcuExecutionFaultKind::ArithmeticUnderflow && fault.invocation_id == 1)
    );
    assert_eq!(
        output.map(f32::to_bits),
        [-1.0_f32, 2.0, -3.0, 42.0, 42.0].map(f32::to_bits)
    );
    tight(&[0.0, -0.0, 1.0], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [
            0x8000_0000,
            0,
            (-1.0_f32).to_bits(),
            42.0_f32.to_bits(),
            42.0_f32.to_bits()
        ]
    );
}

#[path = "f32/f32.rs"]
mod f32;
#[path = "f64/f64.rs"]
mod f64;
#[path = "generic_float/generic_float.rs"]
mod generic_float;
#[path = "integer_fixture.rs"]
mod integers;
#[path = "low_precision/low_precision.rs"]
mod low_precision;

#[test]
#[ignore = "Requires actual Metal discovery and owned session source execution."]
fn annotated_source_accepts_generation_bound_owned_session() {
    use pcu_facade::PcuRuntimeDiscovery;
    let _policy_guard = crate::source_policy_guard();
    let inventory = fusion_pcu_metal::MetalDiscovery::discover().unwrap();
    let mut providers = [pcu_facade::PcuProviderDescriptor {
        id: pcu_facade::PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: pcu_facade::PcuProviderReadiness {
            status: pcu_facade::PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    assert_eq!(inventory.providers(&mut providers).unwrap(), 1);
    let selected = pcu_facade::PcuObjectRef {
        provider: providers[0].id,
        generation: providers[0].generation,
        kind: pcu_facade::PcuObjectKind::Device,
        id: 0,
    };
    let backend = inventory.open_owned_device(selected).unwrap();
    let mut source = negate_prepare::<3, _>(&backend).unwrap();
    drop(backend);
    drop(inventory);
    let mut output = [91.0_f32; 5];
    source(&[0.0, -0.0, f32::from_bits(1)], &mut output).unwrap();
    let expected = [
        0x8000_0000,
        0,
        0x8000_0001,
        91.0_f32.to_bits(),
        91.0_f32.to_bits(),
    ];
    assert_eq!(output.map(f32::to_bits), expected);
    assert!(
        matches!(source(&[1.0, f32::INFINITY, f32::NAN], &mut output), Err(pcu_facade::PcuHostDispatchError::Backend(fusion_pcu_metal::MetalError::Arithmetic(fault))) if fault.invocation_id == 1)
    );
    assert_eq!(output.map(f32::to_bits), expected);
    source(&[1.0, 2.0, 3.0], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [-1.0_f32, -2.0, -3.0, 91.0, 91.0].map(f32::to_bits)
    );
}

#[test]
#[ignore = "Requires actual Metal typed resident execution of annotated source."]
fn annotated_device_neg_keeps_data_resident_and_validates_declared_extent() {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuDeviceBuffer,
        PcuDeviceBufferAllocator,
        PcuMemoryAccess,
        PcuMemoryAllocationRequest,
        PcuMemoryHostAccess,
        PcuMemoryPoolId,
        PcuMemoryProvider,
        PcuOwnedDispatchMemorySession,
        PcuRuntimeDiscovery,
    };
    let _policy_guard = crate::source_policy_guard();
    let inventory = fusion_pcu_metal::MetalDiscovery::discover().unwrap();
    let mut providers = [pcu_facade::PcuProviderDescriptor {
        id: pcu_facade::PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: pcu_facade::PcuProviderReadiness {
            status: pcu_facade::PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    inventory.providers(&mut providers).unwrap();
    let backend = inventory
        .open_owned_device(pcu_facade::PcuObjectRef {
            provider: providers[0].id,
            generation: providers[0].generation,
            kind: pcu_facade::PcuObjectKind::Device,
            id: 0,
        })
        .unwrap();
    let mut provider = backend.memory_provider(PcuMemoryPoolId(10));
    let request = PcuMemoryAllocationRequest {
        pool: PcuMemoryPoolId(10),
        size_bytes: 20,
        alignment_bytes: 4,
        access: PcuMemoryAccess::ReadWrite,
        host_access: PcuMemoryHostAccess::TransferOnly,
        require_device_local: false,
    };
    let mut input = provider.allocate_device_buffer::<f32>(request, 5).unwrap();
    let mut output = provider.allocate_device_buffer::<f32>(request, 5).unwrap();
    let bytes = |words: [u32; 5]| {
        words
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>()
    };
    provider
        .transfer_to(input.resource_mut(), 0, &bytes([0, 0x8000_0000, 1, 77, 88]))
        .unwrap();
    provider
        .transfer_to(output.resource_mut(), 0, &bytes([91; 5]))
        .unwrap();
    let mut source = negate_prepare_device::<3, _>(&backend).unwrap();
    drop(backend);
    source(&input, &mut output).unwrap();
    let mut result = [0; 20];
    provider
        .transfer_from(output.resource(), 0, &mut result)
        .unwrap();
    assert_eq!(
        result.to_vec(),
        bytes([0x8000_0000, 0, 0x8000_0001, 91, 91])
    );
    provider
        .transfer_to(
            input.resource_mut(),
            0,
            &bytes([1.0_f32.to_bits(), 0x7f80_0001, 0x7f80_0000, 77, 88]),
        )
        .unwrap();
    assert!(
        matches!(source(&input, &mut output), Err(fusion_pcu_metal::MetalOwnedDispatchError::Metal(fusion_pcu_metal::MetalError::Arithmetic(fault))) if fault.invocation_id == 1 && fault.kind == pcu_facade::PcuExecutionFaultKind::InvalidFloatingOperand)
    );
    provider
        .transfer_from(output.resource(), 0, &mut result)
        .unwrap();
    assert_eq!(result.to_vec(), bytes([(-1.0_f32).to_bits(), 0, 0, 91, 91]));
    provider
        .transfer_to(input.resource_mut(), 0, &bytes([0, 0x8000_0000, 1, 77, 88]))
        .unwrap();
    source(&input, &mut output).unwrap();
    let short = PcuDeviceBuffer::<f32, _>::new(input.into_resource(), 1);
    assert!(matches!(
        source(&short, &mut output),
        Err(fusion_pcu_metal::MetalOwnedDispatchError::Binding(
            pcu_facade::PcuOwnedDispatchBindingError::BufferTooSmall { .. }
        ))
    ));
    let forged = PcuDeviceBuffer::<f32, _>::new(short.into_resource(), 6);
    assert!(matches!(
        source(&forged, &mut output),
        Err(fusion_pcu_metal::MetalOwnedDispatchError::Binding(
            pcu_facade::PcuOwnedDispatchBindingError::BufferTooSmall { .. }
        ))
    ));
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate_f64<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = N, flag(reject_subnormal_result), crate_path = ::pcu_facade)]
fn negate_f64_tight<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[test]
#[ignore = "Requires actual Metal paired-limb F64 source and resident hardware qualification."]
fn f64_unary_exact_encoding_fault_order_and_resident_tails() {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuFloatUnderflowPolicy,
    };
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut source = negate_f64_prepare::<9, _>(&session).unwrap();
    let bits = [
        0,
        0x8000_0000_0000_0000,
        1,
        0x8000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        1.0_f64.to_bits(),
    ];
    let input = bits.map(f64::from_bits);
    let mut output = [91.0_f64; 11];
    source(&input, &mut output).unwrap();
    let expected: Vec<_> = bits
        .into_iter()
        .map(|value| value ^ 0x8000_0000_0000_0000)
        .chain([91.0_f64.to_bits(); 2])
        .collect();
    assert_eq!(output.map(f64::to_bits).as_slice(), expected);
    let mut invalid = input;
    for encoding in [
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff8_0000_0000_0001,
    ] {
        invalid[3] = f64::from_bits(encoding);
        invalid[8] = f64::INFINITY;
        let Err(pcu_facade::PcuHostDispatchError::Backend(
            fusion_pcu_metal::MetalError::Arithmetic(fault),
        )) = source(&invalid, &mut output)
        else {
            panic!("missing F64 encoding fault");
        };
        assert_eq!(fault.invocation_id, 3);
        assert_eq!(
            fault.kind,
            pcu_facade::PcuExecutionFaultKind::InvalidFloatingOperand
        );
        assert_eq!(output.map(f64::to_bits).as_slice(), expected);
    }
    let mut tight = negate_f64_tight_prepare::<9, _>(&session).unwrap();
    let Err(pcu_facade::PcuHostDispatchError::Backend(fusion_pcu_metal::MetalError::Arithmetic(
        fault,
    ))) = tight(&input, &mut output)
    else {
        panic!("missing F64 subnormal fault");
    };
    assert_eq!(fault.invocation_id, 2);
    assert_eq!(
        fault.kind,
        pcu_facade::PcuExecutionFaultKind::ArithmeticUnderflow
    );
    assert_eq!(output.map(f64::to_bits).as_slice(), expected);
    let word_bytes: Vec<_> = bits.into_iter().flat_map(u64::to_le_bytes).collect();
    let staged = session.upload_bytes(&word_bytes).unwrap();
    let relu = session
        .prepare_f64_relu(PcuFloatUnderflowPolicy::IeeeAfterRounding)
        .unwrap();
    let mut result = [0_u8; 72];
    relu.execute(&staged)
        .unwrap()
        .read_into_bytes(&mut result)
        .unwrap();
    let relu_expected: Vec<_> = bits
        .into_iter()
        .map(|value| if value >> 63 == 0 { value } else { 0 })
        .flat_map(u64::to_le_bytes)
        .collect();
    assert_eq!(result.as_slice(), relu_expected);
    drop((source, tight, session));
    f64_resident(&input, &expected);
}
fn f64_resident(input: &[f64], expected: &[u64]) {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuMemoryPoolId,
        PcuRuntimeDiscovery,
    };
    let mut output = [91.0_f64; 11];
    let discovery = fusion_pcu_metal::MetalDiscovery::discover().unwrap();
    let mut providers = [pcu_facade::PcuProviderDescriptor {
        id: pcu_facade::PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: pcu_facade::PcuProviderReadiness {
            status: pcu_facade::PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    discovery.providers(&mut providers).unwrap();
    let backend = discovery
        .open_owned_device(pcu_facade::PcuObjectRef {
            provider: providers[0].id,
            generation: providers[0].generation,
            kind: pcu_facade::PcuObjectKind::Device,
            id: 0,
        })
        .unwrap();
    let pool = PcuMemoryPoolId(79);
    let input = backend.upload_buffer(pool, input).unwrap();
    let mut resident = backend.upload_buffer(pool, &[91.0_f64; 11]).unwrap();
    let mut device_source = negate_f64_prepare_device::<9, _>(&backend).unwrap();
    device_source(&input, &mut resident).unwrap();
    backend
        .download_buffer(pool, &resident, &mut output)
        .unwrap();
    assert_eq!(output.map(f64::to_bits).as_slice(), expected);
    drop(discovery);
    device_source(&input, &mut resident).unwrap();
}

#[test]
#[ignore = "Requires actual Metal ordinary global paired-limb F64 source routing."]
fn ordinary_global_f64_bits_fault_and_retry() {
    let _policy_guard = crate::source_policy_guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut output = [91.0_f64; 5];
    negate_f64::<3>(&[0.0, -0.0, f64::from_bits(1)], &mut output).unwrap();
    let expected = [
        0x8000_0000_0000_0000,
        0,
        0x8000_0000_0000_0001,
        91.0_f64.to_bits(),
        91.0_f64.to_bits(),
    ];
    assert_eq!(output.map(f64::to_bits), expected);
    let Err(pcu_facade::PcuExecutionError::ArithmeticFault(fault)) =
        negate_f64::<3>(&[1.0, f64::INFINITY, f64::NAN], &mut output)
    else {
        panic!("missing ordinary F64 fault");
    };
    assert_eq!(fault.invocation_id, 1);
    assert_eq!(
        fault.kind,
        pcu_facade::PcuExecutionFaultKind::InvalidFloatingOperand
    );
    assert_eq!(output.map(f64::to_bits), expected);
    negate_f64::<3>(&[1.0, -2.0, 3.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [-1.0_f64, 2.0, -3.0, 91.0, 91.0].map(f64::to_bits)
    );
}

#[path = "integer64/integer64.rs"]
mod integer64;

#[path = "portable/portable.rs"]
mod portable;

#[path = "low_unary/low_unary.rs"]
mod low_unary;

#[path = "native_unary/native_unary.rs"]
mod native_unary;

#[path = "binary_range/binary_range.rs"]
mod binary_range;

#[path = "carrier/carrier.rs"]
mod carrier;

#[path = "portable_unary/portable_unary.rs"]
mod portable_unary;

#[path = "normal_unary/normal_unary.rs"]
mod normal_unary;

#[path = "transport/transport.rs"]
mod transport;

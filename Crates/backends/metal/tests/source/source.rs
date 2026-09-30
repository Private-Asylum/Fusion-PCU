//! Actual annotated identity source against the explicit Metal host authoring seam.
use pcu_facade::pcu;
use fusion_pcu_metal::MetalSession;

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn identity<const N: usize>(input: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[test]
#[ignore = "Requires actual macOS Metal device and runs annotated source on GPU."]
fn annotated_source_current_inputs_and_tail_preservation() {
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

#[path = "integer_fixture.rs"]
mod integers;

#[test]
#[ignore = "Requires actual Metal discovery and owned session source execution."]
fn annotated_source_accepts_generation_bound_owned_session() {
    use pcu_facade::PcuRuntimeDiscovery;
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

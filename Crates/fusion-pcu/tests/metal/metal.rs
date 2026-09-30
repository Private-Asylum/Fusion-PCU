//! Actual ordinary source through the Metal facade and its static resident owner seam.
#![cfg(feature = "metal")]

#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
};
#[rustfmt::skip]
#[cfg(target_os = "macos")]
use fusion_pcu::PcuExecutionFaultKind;
#[rustfmt::skip]
use fusion_pcu::global::{
    configure,
    PcuBackendChoice,
    PcuExecutionPolicy,
};

#[pcu(invocations = 3)]
fn negate(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = 3)]
fn add(lhs: &[u32], rhs: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] + rhs[id];
}

#[pcu(invocations = 3)]
fn negate_f64(input: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[test]
#[cfg(not(target_os = "macos"))]
fn explicit_metal_never_substitutes_cpu_and_portable_policy_rejects_cold() {
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut output = [91.0_f32; 5];
    assert!(matches!(
        negate(&[1.0, 2.0, 3.0], &mut output),
        Err(PcuExecutionError::NoCompatibleInvocationDevice { .. })
    ));
    assert_eq!(output.map(f32::to_bits), [91.0_f32; 5].map(f32::to_bits));
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        numerical_options: fusion_pcu::PcuNumericalOptions {
            reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
            ..fusion_pcu::PcuNumericalOptions::default()
        },
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    assert!(matches!(
        negate(&[1.0, 2.0, 3.0], &mut output),
        Err(PcuExecutionError::UnsupportedNumericalOptions(_))
    ));
    configure(PcuExecutionPolicy::default()).unwrap();
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "Requires actual Metal facade source, mixed resident and policy execution."]
fn native_global_source_mixed_resident_fault_publication_and_policy() {
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuDeviceBuffer,
        PcuMemoryPoolId,
        PcuTensor,
    };
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut host_output = [91.0_f32; 5];
    negate(&[0.0, -0.0, f32::from_bits(1)], &mut host_output).unwrap();
    assert_eq!(
        host_output.map(f32::to_bits),
        [
            0x8000_0000,
            0,
            0x8000_0001,
            91.0_f32.to_bits(),
            91.0_f32.to_bits()
        ]
    );
    let before = host_output.map(f32::to_bits);
    assert!(
        matches!(negate(&[1.0, f32::INFINITY, f32::NAN], &mut host_output), Err(PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id == 1 && fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand)
    );
    assert_eq!(host_output.map(f32::to_bits), before);
    negate(&[1.0, 2.0, 3.0], &mut host_output).unwrap();
    let mut integer_output = [91; 5];
    add(&[1, 2, 3], &[4, 5, 6], &mut integer_output).unwrap();
    assert_eq!(integer_output, [5, 7, 9, 91, 91]);
    assert!(
        matches!(add(&[u32::MAX, 2, 3], &[1, 5, 6], &mut integer_output), Err(PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id == 0)
    );
    assert_eq!(integer_output, [5, 7, 9, 91, 91]);
    let backend = open_backend();
    let buffer = backend
        .upload_buffer(PcuMemoryPoolId(12), &[1.0_f32, 2.0, 3.0, 91.0, 91.0])
        .unwrap();
    let mut resident = PcuTensor::from_device_buffer(backend, buffer, &[5]).unwrap();
    negate(&resident, &mut host_output).unwrap();
    assert_eq!(
        host_output.map(f32::to_bits),
        [-1.0_f32, -2.0, -3.0, 91.0, 91.0].map(f32::to_bits)
    );
    negate(&[0.0, -0.0, f32::from_bits(1)], &mut resident).unwrap();
    let mut resident_result = [0.0; 5];
    resident.read_into(&mut resident_result).unwrap();
    assert_eq!(
        resident_result.map(f32::to_bits),
        [
            0x8000_0000,
            0,
            0x8000_0001,
            91.0_f32.to_bits(),
            91.0_f32.to_bits()
        ]
    );
    assert!(
        matches!(negate(&[1.0, f32::INFINITY, f32::NAN], &mut resident), Err(PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id == 1)
    );
    resident.read_into(&mut resident_result).unwrap();
    assert_eq!(
        resident_result.map(f32::to_bits),
        [
            (-1.0_f32).to_bits(),
            0,
            0,
            91.0_f32.to_bits(),
            91.0_f32.to_bits()
        ]
    );
    negate(&[1.0, 2.0, 3.0], &mut resident).unwrap();
    // A different explicit device never substitutes the retained resident affinity.
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        device: Some(u32::MAX),
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    assert!(matches!(
        negate(&resident, &mut host_output),
        Err(PcuExecutionError::ResidentPolicyConflict)
    ));
    assert!(negate(&[1.0, 2.0, 3.0], &mut host_output).is_err());
    configure(PcuExecutionPolicy::default()).unwrap();
    negate(&[1.0, 2.0, 3.0], &mut host_output).unwrap();
    // Import validates real extent and affinity without transfers or compilation.
    let backend = open_backend();
    let buffer = backend
        .upload_buffer(PcuMemoryPoolId(12), &[1.0_f32; 3])
        .unwrap();
    let forged = PcuDeviceBuffer::new(buffer.into_resource(), 4);
    assert!(PcuTensor::<f32>::from_device_buffer(backend, forged, &[4]).is_err());
    assert_native_cold_rejections();
    configure(PcuExecutionPolicy::default()).unwrap();
}

#[cfg(target_os = "macos")]
fn assert_native_cold_rejections() {
    let mut output = [91.0_f32; 5];
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        numerical_options: fusion_pcu::PcuNumericalOptions {
            reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
            ..fusion_pcu::PcuNumericalOptions::default()
        },
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    assert!(matches!(
        negate(&[1.0, 2.0, 3.0], &mut output),
        Err(PcuExecutionError::UnsupportedNumericalOptions(_))
    ));
    assert_eq!(output.map(f32::to_bits), [91.0_f32; 5].map(f32::to_bits));
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        range_policy: fusion_pcu::PcuRangePolicy::Clamp,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    assert!(matches!(
        negate(&[1.0, 2.0, 3.0], &mut output),
        Err(PcuExecutionError::NoCompatibleInvocationDevice { .. })
    ));
    assert_eq!(output.map(f32::to_bits), [91.0_f32; 5].map(f32::to_bits));
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Metal,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut wide = [91.0_f64; 5];
    assert!(matches!(
        negate_f64(&[1.0, 2.0, 3.0], &mut wide),
        Err(PcuExecutionError::NoCompatibleInvocationDevice { .. })
    ));
    assert_eq!(wide.map(f64::to_bits), [91.0_f64; 5].map(f64::to_bits));
    let backend = open_backend();
    let buffer = backend
        .upload_buffer(fusion_pcu::PcuMemoryPoolId(12), &[1.0_f32; 3])
        .unwrap();
    assert!(
        fusion_pcu::PcuTensor::<f32>::from_device_buffer(open_backend(), buffer, &[3]).is_err()
    );
    let buffer = backend
        .upload_buffer(fusion_pcu::PcuMemoryPoolId(12), &[1.0_f32; 3])
        .unwrap();
    assert!(fusion_pcu::PcuTensor::<f32>::from_device_buffer(backend, buffer, &[2]).is_err());
}

#[cfg(target_os = "macos")]
fn open_backend() -> fusion_pcu::metal::MetalOwnedDispatchBackend {
    use fusion_pcu::PcuRuntimeDiscovery;
    let inventory = fusion_pcu::metal::MetalDiscovery::discover().unwrap();
    let mut providers = [fusion_pcu::PcuProviderDescriptor {
        id: fusion_pcu::PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: fusion_pcu::PcuProviderReadiness {
            status: fusion_pcu::PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    inventory.providers(&mut providers).unwrap();
    inventory
        .open_owned_device(fusion_pcu::PcuObjectRef {
            provider: providers[0].id,
            generation: providers[0].generation,
            kind: fusion_pcu::PcuObjectKind::Device,
            id: 0,
        })
        .unwrap()
}

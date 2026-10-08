//! Actual requested private staging bytes; full host and resident views stay authoritative.
use super::*;
#[rustfmt::skip]
use crate::{
    RocmDiscovery,
    RocmOwnedDispatchBackend,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuObjectKind,
    PcuObjectRef,
    PcuRuntimeDiscovery,
    PcuTargetDescriptor,
    PcuDispatchKernelIr,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[path = "../benches/ordered_float_maps/oracle/oracle.rs"]
mod oracle;
#[path = "../benches/ordered_float_maps/source/source.rs"]
#[allow(dead_code)] // Both genuine annotated entries share the generated helpers used here.
pub(super) mod source;
use oracle::Format;

pub(super) fn selected_device() -> (RocmDiscovery, RocmOwnedDispatchBackend) {
    let discovery = RocmDiscovery::new();
    let invalid = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Device,
        id: 0,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    assert_eq!(discovery.providers(&mut providers).unwrap(), 1);
    let mut targets = [PcuTargetDescriptor {
        reference: invalid,
        name: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    assert_eq!(
        discovery
            .targets(providers[0].id, providers[0].generation, &mut targets)
            .unwrap(),
        1
    );
    let count = discovery.devices(targets[0].reference, &mut []).unwrap();
    assert!(count > 0, "test requires a visible ROCm device");
    let mut devices = vec![
        PcuDeviceDescriptor {
            reference: invalid,
            target: invalid,
            name: "",
            class: PcuDeviceClass::Other,
            vendor: None,
            architecture: None,
            generation: None,
            location: None,
        };
        count
    ];
    discovery
        .devices(targets[0].reference, &mut devices)
        .unwrap();
    let session = RocmOwnedDispatchBackend::open(&discovery, devices[0].reference, 256)
        .expect("open selected ROCm device");
    (discovery, session)
}

fn call<T: Format>(
    prepared: &mut RocmPreparedHostKernel,
    input: &[T],
    stage: &mut [T],
    output: &mut [T],
) -> Result<(), RocmHostKernelError> {
    prepared.call(&mut [
        PcuHostArgument::read_write(PcuBindingRef::new(0, 0), stage),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
        PcuHostArgument::read(PcuBindingRef::new(0, 2), input),
    ])
}
fn slots<T: Format>(prepared: &RocmPreparedHostKernel, input_elements: usize) {
    let width = core::mem::size_of::<T>();
    for (slot, requirement) in prepared.dispatch.binding_schema().iter().enumerate() {
        let output = requirement.target.binding < 2;
        let expected = if output { 65 } else { input_elements } * width;
        assert_eq!(
            prepared.slots[slot].resource.as_ref().unwrap().size_bytes(),
            u64::try_from(expected).unwrap(),
            "{} physical slot {:?}",
            T::LABEL,
            requirement.target
        );
        assert_eq!(
            prepared.slots[slot].fully_written_prefix,
            output.then_some(65 * width)
        );
    }
}
fn values<T: Format>(actual: &[T], expected: &[T]) {
    for (got, want) in actual.iter().zip(expected) {
        assert_eq!(got.bits(), want.bits());
    }
    for value in &actual[expected.len()..] {
        assert_eq!(value.bits(), T::sentinel().bits());
    }
}
fn exercise<T: Format>(backend: &RocmOwnedDispatchBackend, ir: &PcuDispatchKernelIr<'_>) {
    let mut prepared = backend.prepare_host_kernel(ir).unwrap();
    let (mut input, expected_stage, expected_output) = oracle::bank::<T>(65, 0);
    input.resize(67, T::from(T::MAX + 1));
    let mut stage = vec![T::sentinel(); 67];
    let mut output = stage.clone();
    let mut invalid = input.clone();
    invalid[0] = T::from(T::MAX + 1);
    assert!(call(&mut prepared, &invalid, &mut stage, &mut output).is_err());
    assert!(
        stage
            .iter()
            .chain(&output)
            .all(|value| value.bits() == T::sentinel().bits())
    );
    assert!(!prepared.last_call_completion_uncertain());
    assert!(prepared.last_call_may_have_written());
    slots::<T>(&prepared, 67);
    call(&mut prepared, &input, &mut stage, &mut output).unwrap();
    oracle::verify(&expected_stage, &stage);
    oracle::verify(&expected_output, &output);
    slots::<T>(&prepared, 67);

    // Larger host tails do not grow complete-writer private allocations. Read input retains
    // its complete original-view allocation and transfer boundary independently.
    let (next, expected_stage, expected_output) = oracle::bank::<T>(65, 1);
    input[..65].copy_from_slice(&next);
    input.resize(94, T::from(T::MAX + 1));
    stage.resize(82, T::sentinel());
    output.resize(88, T::sentinel());
    call(&mut prepared, &input, &mut stage, &mut output).unwrap();
    values(&stage, &expected_stage);
    values(&output, &expected_output);
    slots::<T>(&prepared, 94);
    let before_stage = stage.clone();
    let before_output = output.clone();
    assert!(call(&mut prepared, &input, &mut stage[..64], &mut output).is_err());
    assert_eq!(stage, before_stage);
    assert_eq!(output, before_output);
    assert!(!prepared.last_call_may_have_written());
    slots::<T>(&prepared, 94);
    #[cfg(feature = "allocation-census")]
    crate::ffi::reset_rocm_api_census();
    call(&mut prepared, &input, &mut stage, &mut output).unwrap();
    #[cfg(feature = "allocation-census")]
    {
        let census = crate::ffi::rocm_api_census();
        assert_eq!(census.device_selections, 1);
        assert_eq!(census.device_synchronizations, 0);
        assert_eq!(census.stream_synchronizations, 0);
        assert_eq!(census.allocations, 0);
        assert!(census.runtime_calls > 1);
    }
    values(&stage, &expected_stage);
    values(&output, &expected_output);
}
fn format<T: Format>(backend: &RocmOwnedDispatchBackend) {
    let bindings = source::direct_bindings::<T>();
    let direct = source::direct_ir::<T, 65>(&bindings).unwrap();
    exercise::<T>(backend, &direct.ir());
    let grid = source::grid_ir::<T, 65>(&bindings).unwrap();
    grid.with_ir(|ir| exercise::<T>(backend, ir));
}
#[test]
#[ignore = "requires the actual authorized AMD GPU"]
fn complete_writers_retain_only_private_initialized_prefix_capacity() {
    let (_discovery, backend) = selected_device();
    format::<PcuF16Bits>(&backend);
    format::<PcuBf16Bits>(&backend);
    format::<PcuF8E4M3FnBits>(&backend);
    format::<PcuF8E5M2Bits>(&backend);
    format::<f32>(&backend);
    format::<f64>(&backend);
}

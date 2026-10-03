//! Actual full-request offer identity, exact original numerical header and honest boundaries.
use super::*;
#[path = "../../scalar_transport/device/device.rs"]
mod device;
#[rustfmt::skip]
use pcu_facade::{
    PcuBindingAccess,
    PcuCostBoundary,
    PcuDispatchKernelIr,
    PcuExecutorId,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuReproducibility,
};
fn check(backend: &PcuVulkanBackend, kernel: &PcuDispatchKernelIr<'_>) {
    let scalar = kernel.bindings[0]
        .binding_type
        .value_type()
        .unwrap()
        .scalar_type();
    let ordinal = pcu_facade::PcuScalarType::ALL
        .iter()
        .position(|ty| *ty == scalar)
        .unwrap();
    let request = PcuImplementationRequest {
        device: backend.device_identity().unwrap(),
        executor: PcuExecutorId(0),
        boundary: PcuCostBoundary::Host,
        operation: kernel,
        requirements: kernel.numerical_requirements,
    };
    let mut offers = [None];
    assert_eq!(
        backend
            .implementation_offers(&request, &mut offers)
            .unwrap(),
        1
    );
    let offer = offers[0].unwrap();
    assert_eq!(
        offer.implementation.local_id,
        19456 + u32::try_from(ordinal).unwrap()
    );
    assert_eq!(offer.implementation.revision, 1);
    assert_eq!(offer.implementation.device, request.device);
    assert_eq!(offer.implementation.executor, request.executor);
    assert_eq!(offer.requirements, request.requirements);
    assert_eq!(backend.implementation_offers(&request, &mut []).unwrap(), 1);
    assert!(matches!(
        backend.prepare_host_kernel(kernel).unwrap(),
        Prepared::OrderedTransport(_)
    ));
    let mut mismatch_requirements = request.requirements;
    mismatch_requirements.float_underflow = match request.requirements.float_underflow {
        pcu_facade::PcuFloatUnderflowPolicy::AllowGradualUnderflow => {
            pcu_facade::PcuFloatUnderflowPolicy::RejectSubnormalResult
        }
        _ => pcu_facade::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    };
    let mismatch = PcuImplementationRequest {
        requirements: mismatch_requirements,
        ..request
    };
    assert_eq!(
        backend.implementation_offers(&mismatch, &mut []).unwrap(),
        0
    );
    let mut portable = *kernel;
    portable
        .numerical_requirements
        .numerical_options
        .reproducibility = PcuReproducibility::PortableV1;
    let changed = PcuImplementationRequest {
        operation: &portable,
        requirements: portable.numerical_requirements,
        ..request
    };
    assert_eq!(backend.implementation_offers(&changed, &mut []).unwrap(), 0);
    assert!(backend.prepare_host_kernel(&portable).is_err());
    let mut bindings = kernel.bindings.to_vec();
    bindings[2].access = PcuBindingAccess::WriteOnly;
    let mut write_only = *kernel;
    write_only.bindings = &bindings;
    let changed = PcuImplementationRequest {
        operation: &write_only,
        ..request
    };
    assert_eq!(backend.implementation_offers(&changed, &mut []).unwrap(), 0);
    assert!(backend.prepare_host_kernel(&write_only).is_err());
}
fn width<T: PcuScalar>(backend: &PcuVulkanBackend) {
    policy::each(|requirements| {
        graph::with(T::TYPE, 47, false, requirements, |kernel| {
            check(backend, kernel);
        });
        graph::with(T::TYPE, 47, true, requirements, |kernel| {
            check(backend, kernel);
        });
    });
}
#[test]
#[ignore = "requires actual Vulkan device for ordinary offers and native cold preparation"]
fn ordinary_twenty_two_offers_preserve_full_tuple_and_refuse_header_mismatch() {
    let (backend, _) = device::selected();
    carriers!(width, &backend);
}

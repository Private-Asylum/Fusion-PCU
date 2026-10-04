//! Actual inventory exact host-only offers, cold numerical envelope and policy identity.
use super::{graph, Policy, Range};
use fusion_pcu::{
    PcuCostBoundary, PcuDeviceIdentity, PcuDispatchCheckedFloatConversion, PcuExecutorId,
    PcuImplementationOffers, PcuImplementationRequest, PcuReproducibility,
};
use fusion_pcu::PcuRuntimeDiscovery;
#[test]
#[ignore = "Requires actual Apple native discovery and qualified conversion backend."]
fn conversion_exact_host_offers_preserve_policy_identity_and_refuse_other_boundaries() {
    let discovery = crate::MetalDiscovery::discover().unwrap();
    let device = device_identity(&discovery);
    verify_source_floor(&discovery, device);
    let mut identities = std::collections::BTreeSet::new();
    for direction in [
        PcuDispatchCheckedFloatConversion::F32ToF64,
        PcuDispatchCheckedFloatConversion::F64ToF32,
    ] {
        for range in [Range::Reject, Range::Clamp] {
            for policy in [
                Policy::IeeeAfterRounding,
                Policy::RejectSubnormalResult,
                Policy::AllowGradualUnderflow,
            ] {
                for broadcast in [false, true] {
                    graph::with(direction, 3, range, policy, broadcast, false, |kernel| {
                        let request = PcuImplementationRequest::for_dispatch(
                            device,
                            PcuExecutorId(0),
                            PcuCostBoundary::Host,
                            kernel,
                        );
                        let mut offers = [None];
                        assert_eq!(
                            discovery
                                .implementation_offers(&request, &mut offers)
                                .unwrap(),
                            1
                        );
                        let offer = offers[0].unwrap();
                        assert_eq!(offer.requirements, kernel.numerical_requirements);
                        assert_eq!(
                            offer.cost,
                            fusion_pcu::PcuImplementationCost::unknown(PcuCostBoundary::Host)
                        );
                        assert_eq!(offer.workspace_bytes, None);
                        assert!(identities.insert(offer.implementation.local_id));
                        assert_eq!(
                            discovery.implementation_offers(&request, &mut []).unwrap(),
                            1
                        );
                        for boundary in [
                            PcuCostBoundary::Resident,
                            PcuCostBoundary::HostInputsResidentOutput,
                            PcuCostBoundary::MixedInputsResidentOutput,
                        ] {
                            let other = PcuImplementationRequest::for_dispatch(
                                device,
                                PcuExecutorId(0),
                                boundary,
                                kernel,
                            );
                            assert_eq!(
                                discovery
                                    .implementation_offers(&other, &mut offers)
                                    .unwrap(),
                                0
                            );
                            assert!(offers[0].is_none());
                        }
                        let mut portable = *kernel;
                        portable
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                        let other = PcuImplementationRequest::for_dispatch(
                            device,
                            PcuExecutorId(0),
                            PcuCostBoundary::Host,
                            &portable,
                        );
                        assert_eq!(
                            discovery
                                .implementation_offers(&other, &mut offers)
                                .unwrap(),
                            0
                        );
                        let mut wrong = request;
                        wrong.requirements.range_policy = if range == Range::Reject {
                            Range::Clamp
                        } else {
                            Range::Reject
                        };
                        assert!(
                            discovery
                                .implementation_offers(&wrong, &mut offers)
                                .is_err()
                        );
                    });
                }
            }
        }
    }
    assert_eq!(identities.len(), 24);
}

fn device_identity(discovery: &crate::MetalDiscovery) -> PcuDeviceIdentity {
    let readiness = fusion_pcu::PcuProviderReadiness {
        status: fusion_pcu::PcuProviderStatus::Unavailable,
        reason: None,
    };
    let empty = fusion_pcu::PcuObjectRef {
        provider: fusion_pcu::PcuProviderId(0),
        generation: 0,
        kind: fusion_pcu::PcuObjectKind::Device,
        id: 0,
    };
    let mut providers = [fusion_pcu::PcuProviderDescriptor {
        id: empty.provider,
        generation: 0,
        backend: "",
        readiness,
    }; 1];
    discovery.providers(&mut providers).unwrap();
    let mut targets = [fusion_pcu::PcuTargetDescriptor {
        reference: empty,
        name: "",
        readiness,
    }; 1];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .unwrap();
    let mut devices = [fusion_pcu::PcuDeviceDescriptor {
        reference: empty,
        target: empty,
        name: "",
        class: fusion_pcu::PcuDeviceClass::Gpu,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }; 1];
    discovery
        .devices(targets[0].reference, &mut devices)
        .unwrap();
    PcuDeviceIdentity::from_device_ref(devices[0].reference).unwrap()
}

fn verify_source_floor(discovery: &crate::MetalDiscovery, device: PcuDeviceIdentity) {
    let snapshot = discovery
        .device_capabilities(fusion_pcu::PcuObjectRef {
            provider: device.provider(),
            generation: device.generation(),
            kind: fusion_pcu::PcuObjectKind::Device,
            id: device.device_id(),
        })
        .unwrap();
    assert!(
        snapshot
            .support
            .dispatch_support
            .instructions
            .direct
            .contains(fusion_pcu::PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
    );
    assert!(
        !crate::owned_dispatch::discovery_support()
            .dispatch_support
            .instructions
            .direct
            .contains(fusion_pcu::PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
    );
}

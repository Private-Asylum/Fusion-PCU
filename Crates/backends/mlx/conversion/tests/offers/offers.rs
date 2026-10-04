//! Actual inventory exact host-only offers, cold numerical envelope and policy identity.
use super::{graph, Policy, Range};
use fusion_pcu::{
    PcuCostBoundary, PcuDeviceIdentity, PcuDispatchCheckedFloatConversion, PcuExecutorId,
    PcuImplementationOffers, PcuImplementationRequest, PcuReproducibility,
};
#[test]
#[ignore = "Requires actual Apple native discovery and qualified conversion backend."]
fn conversion_exact_host_offers_preserve_policy_identity_and_refuse_other_boundaries() {
    let discovery = crate::MlxDiscovery::discover_default().unwrap();
    let device =
        PcuDeviceIdentity::from_device_ref(discovery.device_reference(0).unwrap()).unwrap();
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

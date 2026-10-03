//! Actual discovery returns exact unary family revision without claiming resident cost evidence.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxDiscovery,
    MLX_EXECUTOR,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuDeviceIdentity,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuCostBoundary,
    PcuImplementationCost,
    PcuScalarType,
    PcuReproducibility,
};
#[rustfmt::skip]
use super::{
    Sample,
    Policy,
};
#[rustfmt::skip]
use super::super::{
    graph,
    Op,
    Range,
};
fn format<T: Sample>(discovery: &MlxDiscovery) {
    let device =
        PcuDeviceIdentity::from_device_ref(discovery.device_reference(0).unwrap()).unwrap();
    for op in [Op::Neg, Op::Relu] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::RejectSubnormalResult,
            Policy::AllowGradualUnderflow,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        graph::fixture_profile::<T, _>(
                            3,
                            op,
                            policy,
                            range,
                            grid,
                            broadcast,
                            |ir| {
                                let request = PcuImplementationRequest {
                                    device,
                                    executor: MLX_EXECUTOR,
                                    requirements: ir.numerical_requirements,
                                    boundary: PcuCostBoundary::Host,
                                    operation: ir,
                                };
                                let mut offers = [None];
                                assert_eq!(
                                    discovery
                                        .implementation_offers(&request, &mut offers)
                                        .unwrap(),
                                    1
                                );
                                let offer = offers[0].unwrap();
                                offer.validate_request(&request).unwrap();
                                assert_eq!(
                                    offer.implementation.revision,
                                    if matches!(T::TYPE, PcuScalarType::F32 | PcuScalarType::F64) {
                                        0x0003_0020_0003_0700
                                    } else {
                                        0x0003_0020_0003_0100
                                    }
                                );
                                assert_eq!(offer.workspace_bytes, None);
                                assert_eq!(
                                    offer.cost,
                                    PcuImplementationCost::unknown(PcuCostBoundary::Host)
                                );
                                let resident = PcuImplementationRequest {
                                    boundary: PcuCostBoundary::Resident,
                                    ..request
                                };
                                assert_eq!(
                                    discovery
                                        .implementation_offers(&resident, &mut offers)
                                        .unwrap(),
                                    0
                                );
                                let mut portable = *ir;
                                portable
                                    .numerical_requirements
                                    .numerical_options
                                    .reproducibility = PcuReproducibility::PortableV1;
                                let request = PcuImplementationRequest {
                                    operation: &portable,
                                    requirements: portable.numerical_requirements,
                                    ..request
                                };
                                assert_eq!(
                                    discovery
                                        .implementation_offers(&request, &mut offers)
                                        .unwrap(),
                                    1
                                );
                                let portable_offer = offers[0].unwrap();
                                portable_offer.validate_request(&request).unwrap();
                                assert_eq!(portable_offer.requirements, request.requirements);
                                assert_eq!(
                                    portable_offer.implementation.revision,
                                    0x0003_0020_0003_1400
                                );
                                assert_ne!(
                                    portable_offer.implementation.local_id,
                                    offer.implementation.local_id
                                );
                            },
                        );
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual pinned MLX discovery and exact six-format unary offers."]
fn six_format_unary_detached_inventory_offers() {
    let discovery = MlxDiscovery::discover_default().unwrap();
    format::<super::PcuF16Bits>(&discovery);
    format::<super::PcuBf16Bits>(&discovery);
    format::<super::PcuF8E4M3FnBits>(&discovery);
    format::<super::PcuF8E5M2Bits>(&discovery);
    format::<f32>(&discovery);
    format::<f64>(&discovery);
}

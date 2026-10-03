//! Native inventory offers and the truthful static Session unary/binary prepared aggregate.
#[rustfmt::skip]
use pcu_facade::{
    PcuDeviceActivation,
    PcuDeviceIdentity,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationCost,
    PcuCostBoundary,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuHostArgument,
    PcuBindingRef,
    PcuReproducibility,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxDiscovery,
    MlxPreparedDispatchKernel,
    MLX_EXECUTOR,
};
#[rustfmt::skip]
use super::{
    Sample,
    Op,
    Policy,
    Range,
    graph,
    source,
};

const fn revision(scalar: pcu_facade::PcuScalarType) -> u64 {
    if matches!(
        scalar,
        pcu_facade::PcuScalarType::F32 | pcu_facade::PcuScalarType::F64
    ) {
        0x0003_0020_0003_0600
    } else {
        0x0003_0020_0003_0200
    }
}

fn qualify<T: Sample>(discovery: &MlxDiscovery) {
    let reference = discovery.device_reference(0).unwrap();
    let device = PcuDeviceIdentity::from_device_ref(reference).unwrap();
    let session = discovery.open_device(reference).unwrap();
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
        let expected = match op {
            Op::Add => T::value(6.0),
            Op::Sub => T::value(-2.0),
            Op::Mul => T::value(8.0),
            Op::Div => T::value(0.5),
        };
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::RejectSubnormalResult,
            Policy::AllowGradualUnderflow,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                graph::fixture_profile::<T, _>(3, op, policy, range, false, [false; 2], |ir| {
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
                    assert_eq!(offer.implementation.revision, revision(T::TYPE));
                    assert_eq!(offer.workspace_bytes, None);
                    assert_eq!(
                        offer.cost,
                        PcuImplementationCost::unknown(PcuCostBoundary::Host)
                    );
                    let resident_request = PcuImplementationRequest {
                        device,
                        executor: MLX_EXECUTOR,
                        requirements: ir.numerical_requirements,
                        boundary: PcuCostBoundary::Resident,
                        operation: ir,
                    };
                    assert_eq!(
                        discovery
                            .implementation_offers(&resident_request, &mut offers)
                            .unwrap(),
                        0
                    );
                    assert_eq!(offers, [None]);
                    let mut portable = *ir;
                    portable
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    let portable_request = PcuImplementationRequest {
                        operation: &portable,
                        requirements: portable.numerical_requirements,
                        ..request
                    };
                    assert_eq!(
                        discovery
                            .implementation_offers(&portable_request, &mut offers)
                            .unwrap(),
                        0
                    );
                    let mut prepared = session.prepare_host_kernel(ir).unwrap();
                    assert!(matches!(prepared, MlxPreparedDispatchKernel::Binary(_)));
                    assert_eq!(
                        prepared.input_bindings(),
                        [PcuBindingRef::new(2, 3), PcuBindingRef::new(3, 2)]
                    );
                    let left = [T::value(2.0); 3];
                    let right = [T::value(4.0); 3];
                    let sentinel = T::from_bits(T::MAX);
                    let mut output = [sentinel; 5];
                    prepared
                        .call(&mut [
                            PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                            PcuHostArgument::read(PcuBindingRef::new(3, 2), &right),
                            PcuHostArgument::read(PcuBindingRef::new(2, 3), &left),
                        ])
                        .unwrap();
                    assert_eq!(&output[..3], &[expected; 3]);
                    assert_eq!(&output[3..], &[sentinel; 2]);
                });
            }
        }
    }
    let mut output = [T::from_bits(T::MAX); 5];
    source::repeated_prepare::<T, 3, _>(&session).unwrap()(&[T::value(2.0); 3], &[], &mut output)
        .unwrap();
    assert_eq!(&output[..3], &[T::value(4.0); 3]);
}

#[test]
#[ignore = "Requires actual MLX discovery offers and retained Session GPU dispatch aggregate."]
fn binary_inventory_and_session_aggregate() {
    let discovery = MlxDiscovery::discover_default().unwrap();
    qualify::<super::PcuF16Bits>(&discovery);
    qualify::<super::PcuBf16Bits>(&discovery);
    qualify::<super::PcuF8E4M3FnBits>(&discovery);
    qualify::<super::PcuF8E5M2Bits>(&discovery);
    qualify::<f32>(&discovery);
    qualify::<f64>(&discovery);
}

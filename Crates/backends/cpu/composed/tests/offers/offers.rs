//! Full request identity and private workspace for the new scalar realization only.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuExecutorId,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuNumericalMode,
    PcuObjectKind,
    PcuObjectRef,
    PcuPrecisionPolicy,
    PcuProviderId,
};
#[test]
fn full_tuple_and_workspace_are_exact_and_portable_remains_closed() {
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 1,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let executor = PcuExecutorId(0);
    let offers = crate::PcuCpuHostOffers::new(crate::PcuCpuHostBackend::scalar(), device, executor);
    let bindings = composed_bindings::<f32>();
    composed_ir::<f32>(&bindings).unwrap().with_ir(|base| {
        for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            for compound_arithmetic in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    let mut kernel = *base;
                    kernel.numerical_requirements.numerical_mode = numerical_mode;
                    kernel
                        .numerical_requirements
                        .numerical_options
                        .compound_arithmetic = compound_arithmetic;
                    kernel.numerical_requirements.numerical_options.precision = precision;
                    let request = PcuImplementationRequest {
                        device,
                        executor,
                        operation: &kernel,
                        requirements: kernel.numerical_requirements,
                        boundary: PcuCostBoundary::Host,
                    };
                    let mut slots = [None];
                    assert_eq!(offers.implementation_offers(&request, &mut slots), Ok(1));
                    let offer = slots[0].unwrap();
                    offer.validate_request(&request).unwrap();
                    assert_eq!(offer.implementation.local_id, 17412);
                    assert_eq!(offer.implementation.revision, 1);
                    assert_eq!(offer.workspace_bytes, Some(16));
                    let mut wrong = request.requirements;
                    wrong.numerical_mode = if numerical_mode == PcuNumericalMode::Boundary {
                        PcuNumericalMode::Strict
                    } else {
                        PcuNumericalMode::Boundary
                    };
                    assert_eq!(
                        offers.implementation_offers(
                            &PcuImplementationRequest {
                                requirements: wrong,
                                ..request
                            },
                            &mut []
                        ),
                        Ok(0)
                    );
                    assert_eq!(
                        offers.implementation_offers(
                            &PcuImplementationRequest {
                                boundary: PcuCostBoundary::Resident,
                                ..request
                            },
                            &mut []
                        ),
                        Ok(0)
                    );
                    let mut portable = kernel;
                    portable
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    assert_eq!(
                        offers.implementation_offers(
                            &PcuImplementationRequest {
                                operation: &portable,
                                requirements: portable.numerical_requirements,
                                ..request
                            },
                            &mut []
                        ),
                        Ok(0)
                    );
                }
            }
        }
    });
}

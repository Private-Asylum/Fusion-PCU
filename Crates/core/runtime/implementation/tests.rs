use super::*;
#[rustfmt::skip]
use crate::{
    PcuCompoundArithmeticPolicy,
    PcuObjectKind,
    PcuObjectRef,
    PcuPrecisionPolicy,
    PcuProviderId,
    PcuReproducibility,
};

fn identity(generation: u64) -> PcuImplementationId {
    PcuImplementationId {
        device: PcuDeviceIdentity::from_device_ref(PcuObjectRef {
            provider: PcuProviderId(11),
            generation,
            kind: PcuObjectKind::Device,
            id: 3,
        })
        .unwrap(),
        executor: PcuExecutorId(2),
        local_id: 9,
        revision: 4,
    }
}

fn request(operation: &[u32]) -> PcuImplementationRequest<'_, [u32]> {
    PcuImplementationRequest {
        device: identity(7).device,
        executor: identity(7).executor,
        requirements: PcuImplementationRequirements::default(),
        boundary: PcuCostBoundary::Resident,
        operation,
    }
}

fn offer() -> PcuImplementationOffer {
    PcuImplementationOffer {
        implementation: identity(7),
        kind: PcuImplementationMechanism::NativeKernel,
        requirements: PcuImplementationRequirements::default(),
        workspace_bytes: Some(0),
        cost: PcuImplementationCost::unknown(PcuCostBoundary::Resident),
    }
}

#[test]
fn stale_device_and_other_executor_offers_do_not_validate() {
    let operation = [32, 64];
    let mut query = request(&operation);
    assert_eq!(offer().validate_request(&query), Ok(()));
    query.device = identity(8).device;
    assert_eq!(
        offer().validate_request(&query),
        Err(PcuImplementationOfferMismatch::Device)
    );
    query.device = identity(7).device;
    query.executor = PcuExecutorId(3);
    assert_eq!(
        offer().validate_request(&query),
        Err(PcuImplementationOfferMismatch::Executor)
    );
}

#[test]
fn every_numerical_axis_and_boundary_stays_part_of_admission() {
    let operation = [32, 64];
    let original = request(&operation);
    let mut changed = original.requirements;
    changed.numerical_mode = PcuNumericalMode::Strict;
    let mut policies = [original.requirements; 6];
    policies[0] = changed;
    policies[1].numerical_options.compound_arithmetic = PcuCompoundArithmeticPolicy::BackendDefined;
    policies[2].numerical_options.precision = PcuPrecisionPolicy::BackendOptimized;
    policies[3].numerical_options.reproducibility = PcuReproducibility::PortableV1;
    policies[4].float_underflow = PcuFloatUnderflowPolicy::AllowGradualUnderflow;
    policies[5].range_policy = PcuRangePolicy::Clamp;
    for requirements in policies {
        let query = PcuImplementationRequest {
            requirements,
            ..request(&operation)
        };
        assert_eq!(
            offer().validate_request(&query),
            Err(PcuImplementationOfferMismatch::Requirements)
        );
    }
    // A benchmark ending at escaped-owner publication cannot reuse a full
    // host-readback estimate, nor a resident estimate that omitted staging.
    let boundaries = [
        PcuCostBoundary::Resident,
        PcuCostBoundary::Host,
        PcuCostBoundary::HostInputsResidentOutput,
        PcuCostBoundary::MixedInputsResidentOutput,
    ];
    for offered in boundaries {
        let mut candidate = offer();
        candidate.cost = PcuImplementationCost::unknown(offered);
        for required in boundaries {
            let query = PcuImplementationRequest {
                boundary: required,
                ..request(&operation)
            };
            assert_eq!(
                candidate.validate_request(&query),
                if offered == required {
                    Ok(())
                } else {
                    Err(PcuImplementationOfferMismatch::Boundary)
                },
            );
        }
    }
}

#[test]
fn unknown_cost_and_workspace_are_not_zero_and_provenance_is_retained() {
    let mut candidate = offer();
    assert_eq!(candidate.workspace_bytes, Some(0));
    candidate.workspace_bytes = None;
    assert_ne!(candidate.workspace_bytes, Some(0));
    assert_eq!(candidate.cost.completion, None);
    let model = PcuDurationEstimate {
        nanoseconds: 0,
        provenance: PcuCostProvenance::Model { revision: 7 },
    };
    candidate.cost.completion = Some(model);
    assert_ne!(candidate.cost.completion, None);
    let measured = PcuDurationEstimate {
        provenance: PcuCostProvenance::Measured { calibration: 7 },
        ..model
    };
    assert_ne!(model, measured);
}

struct Provider;

impl PcuImplementationOffers<[u32]> for Provider {
    type Error = PcuImplementationOfferMismatch;

    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, [u32]>,
        output: &mut [Option<PcuImplementationOffer>],
    ) -> Result<usize, Self::Error> {
        offer().validate_request(request)?;
        if request.operation != [32, 64] {
            return Ok(0);
        }
        for (index, slot) in output.iter_mut().take(2).enumerate() {
            let mut candidate = offer();
            candidate.implementation.local_id += u32::try_from(index).unwrap();
            *slot = Some(candidate);
        }
        Ok(2)
    }
}

#[test]
fn bounded_enumeration_reports_total_and_requires_concrete_operation_admission() {
    let operation = [32, 64];
    let query = request(&operation);
    assert_eq!(Provider.implementation_offers(&query, &mut []), Ok(2));
    let mut truncated = [None];
    assert_eq!(
        Provider.implementation_offers(&query, &mut truncated),
        Ok(2)
    );
    assert_eq!(truncated[0], Some(offer()));
    let mut all = [None; 3];
    assert_eq!(Provider.implementation_offers(&query, &mut all), Ok(2));
    assert_ne!(all[0], all[1]);
    assert_eq!(all[2], None);
    let unsupported = request(&[64, 32]);
    assert_eq!(offer().validate_request(&unsupported), Ok(()));
    assert_eq!(
        Provider.implementation_offers(&unsupported, &mut all),
        Ok(0)
    );
}

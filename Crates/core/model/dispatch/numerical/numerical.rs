//! The numerical tuple belongs to kernel identity, including owned grid-stride views.
use super::super::PcuDispatchKernelBuilder;
#[rustfmt::skip]
use crate::{
    PcuCompoundArithmeticPolicy,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchDataOp,
    PcuDispatchValueId,
    PcuExecutorId,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuObjectKind,
    PcuObjectRef,
    PcuPrecisionPolicy,
    PcuProviderId,
    PcuRangePolicy,
    PcuReproducibility,
};

#[test]
fn canonical_constant_is_the_checked_default() {
    assert_eq!(
        PcuImplementationRequirements::DEFAULT,
        PcuImplementationRequirements::default()
    );
    let builder = PcuDispatchKernelBuilder::<1>::new(1, "default", [1, 1, 1]);
    assert_eq!(
        builder.ir().numerical_requirements,
        PcuImplementationRequirements::DEFAULT
    );
}

#[test]
fn every_numerical_axis_distinguishes_dispatch_identity() {
    let original = PcuDispatchKernelBuilder::<1>::new(1, "same", [1, 1, 1]);
    let mut variants = [PcuImplementationRequirements::DEFAULT; 6];
    variants[0].numerical_mode = PcuNumericalMode::Strict;
    variants[1].numerical_options.compound_arithmetic = PcuCompoundArithmeticPolicy::BackendDefined;
    variants[2].numerical_options.precision = PcuPrecisionPolicy::BackendOptimized;
    variants[3].numerical_options.reproducibility = PcuReproducibility::PortableV1;
    variants[4].float_underflow = PcuFloatUnderflowPolicy::AllowGradualUnderflow;
    variants[5].range_policy = PcuRangePolicy::Clamp;
    for requirements in variants {
        let changed = original.with_numerical_requirements(requirements);
        assert_ne!(original.ir(), changed.ir());
        assert_eq!(changed.ir().numerical_requirements, requirements);
    }
}

#[test]
fn grid_views_and_borrowed_requests_retain_the_exact_tuple() {
    let mut requirements = PcuImplementationRequirements::DEFAULT;
    requirements.numerical_mode = PcuNumericalMode::Strict;
    requirements.float_underflow = PcuFloatUnderflowPolicy::RejectSubnormalResult;
    let body = PcuDispatchKernelBuilder::<1>::new(1, "grid", [3, 1, 1])
        .with_numerical_requirements(requirements)
        .with_data_op(PcuDispatchDataOp::Constant {
            result: PcuDispatchValueId(0),
            value: crate::PcuParameterValue::from_f32_bits(0),
        })
        .unwrap();
    let grid = crate::model::PcuGridStrideKernelBuilder::new(body, 7).unwrap();
    grid.with_ir(|kernel| {
        assert_eq!(kernel.numerical_requirements, requirements);
        let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
            provider: PcuProviderId(1),
            generation: 2,
            kind: PcuObjectKind::Device,
            id: 3,
        })
        .unwrap();
        let request = PcuImplementationRequest::for_dispatch(
            device,
            PcuExecutorId(4),
            PcuCostBoundary::Host,
            kernel,
        );
        assert_eq!(request.requirements, requirements);
        assert!(core::ptr::eq(request.operation, kernel));
    });
    grid.with_numerical_requirements(PcuImplementationRequirements::DEFAULT)
        .with_ir(|kernel| {
            assert_eq!(
                kernel.numerical_requirements,
                PcuImplementationRequirements::DEFAULT
            );
        });
}

//! Exact ordinary implementation tuple; no resident or Portable composed offer.
use super::*;
#[path = "../../scalar_transport/device/device.rs"]
pub mod device;
#[rustfmt::skip]
use pcu_facade::{
    PcuCostBoundary,
    PcuDispatchKernelIr,
    PcuExecutorId,
    PcuHostKernelBackend,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuReproducibility,
};

fn verify(backend: &PcuVulkanBackend, original: &PcuDispatchKernelIr<'_>, ordinal: u32) {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                let mut kernel = *original;
                kernel.numerical_requirements.numerical_mode = mode;
                kernel
                    .numerical_requirements
                    .numerical_options
                    .compound_arithmetic = compound;
                kernel.numerical_requirements.numerical_options.precision = precision;
                let request = PcuImplementationRequest {
                    device: backend.device_identity().unwrap(),
                    executor: PcuExecutorId(0),
                    boundary: PcuCostBoundary::Host,
                    operation: &kernel,
                    requirements: kernel.numerical_requirements,
                };
                assert_eq!(backend.implementation_offers(&request, &mut []).unwrap(), 1);
                let mut offers = [None];
                assert_eq!(
                    backend
                        .implementation_offers(&request, &mut offers)
                        .unwrap(),
                    1
                );
                let offer = offers[0].unwrap();
                assert_eq!(offer.implementation.local_id, 17664 + ordinal);
                assert_eq!(offer.implementation.revision, 1);
                assert_eq!(offer.implementation.device, request.device);
                assert_eq!(offer.implementation.executor, request.executor);
                assert_eq!(offer.requirements, request.requirements);
                assert!(matches!(
                    backend.prepare_host_kernel(&kernel).unwrap(),
                    PcuVulkanPreparedHost::Composed(_)
                ));
                let mut mismatch = PcuImplementationRequest {
                    device: request.device,
                    executor: request.executor,
                    boundary: request.boundary,
                    operation: &kernel,
                    requirements: request.requirements,
                };
                mismatch.requirements.range_policy =
                    if request.requirements.range_policy == Range::Reject {
                        Range::Clamp
                    } else {
                        Range::Reject
                    };
                assert_eq!(
                    backend.implementation_offers(&mismatch, &mut []).unwrap(),
                    0
                );
                let mut portable_kernel = kernel;
                portable_kernel
                    .numerical_requirements
                    .numerical_options
                    .reproducibility = PcuReproducibility::PortableV1;
                let portable = PcuImplementationRequest {
                    device: request.device,
                    executor: request.executor,
                    boundary: request.boundary,
                    operation: &portable_kernel,
                    requirements: portable_kernel.numerical_requirements,
                };
                assert_eq!(
                    backend.implementation_offers(&portable, &mut []).unwrap(),
                    0
                );
                assert!(backend.prepare_host_kernel(&portable_kernel).is_err());
            }
        }
    }
}

fn width<T: Format>(backend: &PcuVulkanBackend, ordinal: u32) {
    macro_rules! profile {
        ($bindings:ident, $ir:ident) => {
            let bindings = source::$bindings::<T>();
            source::$ir::<T, 65>(&bindings)
                .unwrap()
                .with_ir(|kernel| verify(backend, kernel, ordinal));
        };
    }
    profile!(reject_ieee_bindings, reject_ieee_ir);
    profile!(reject_gradual_bindings, reject_gradual_ir);
    profile!(reject_tight_bindings, reject_tight_ir);
    profile!(clamp_ieee_bindings, clamp_ieee_ir);
    profile!(clamp_gradual_bindings, clamp_gradual_ir);
    profile!(clamp_tight_bindings, clamp_tight_ir);
}

#[test]
#[ignore = "requires Vulkan hardware for exact ordinary offers and cold native prepare"]
fn six_format_ordinary_composed_full_tuple_and_header_refusals() {
    let (backend, _) = device::selected();
    width::<pcu_facade::PcuF16Bits>(&backend, 0);
    width::<pcu_facade::PcuBf16Bits>(&backend, 1);
    width::<pcu_facade::PcuF8E4M3FnBits>(&backend, 2);
    width::<pcu_facade::PcuF8E5M2Bits>(&backend, 3);
    width::<f32>(&backend, 4);
    width::<f64>(&backend, 5);
}

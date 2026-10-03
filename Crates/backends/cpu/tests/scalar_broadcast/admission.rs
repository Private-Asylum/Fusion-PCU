//! Exact closed offers, immutable cold schema and publication preflight.
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuHostBackend,PcuCpuIdentity,PcuCpuHostOffers};
#[rustfmt::skip]
use pcu_facade::{PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuCostBoundary,PcuDeviceIdentity,PcuExecutorId,PcuProviderId,PcuObjectRef,PcuObjectKind,PcuImplementationOffers,PcuImplementationRequest,PcuDispatchKernelIr,PcuNumericalMode,PcuReproducibility,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy};
#[path = "../prepared_identity/source/source.rs"]
mod dense;
pub fn check<T: super::Sample>() {
    let bindings = super::source::direct_bindings::<T>();
    let builder = super::source::direct_ir::<T, 7>(&bindings).unwrap();
    let kernel = builder.ir();
    let dense_bindings = dense::copy_bindings::<T>();
    let dense_builder = dense::copy_ir::<T, 7>(&dense_bindings).unwrap();
    let dense_kernel = dense_builder.ir();
    assert_eq!(
        PcuCpuIdentity
            .prepare_host_kernel(&dense_kernel)
            .unwrap()
            .local_id(),
        64 + T::TYPE as u32
    );
    let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), identity, PcuExecutorId(0));
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                let mut requirements = kernel.numerical_requirements;
                requirements.numerical_mode = mode;
                requirements.numerical_options.compound_arithmetic = compound;
                requirements.numerical_options.precision = precision;
                let kernel = PcuDispatchKernelIr {
                    numerical_requirements: requirements,
                    ..kernel
                };
                let request = PcuImplementationRequest {
                    device: identity,
                    executor: PcuExecutorId(0),
                    operation: &kernel,
                    requirements,
                    boundary: PcuCostBoundary::Host,
                };
                let mut output = [None];
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
                let offer = output[0].unwrap();
                assert_eq!(offer.implementation.local_id, 384 + T::TYPE as u32);
                assert_eq!(offer.implementation.revision, 1);
                assert_eq!(offer.workspace_bytes, Some(0));
                let mut request = request;
                request.requirements.numerical_mode = if mode == PcuNumericalMode::Strict {
                    PcuNumericalMode::Boundary
                } else {
                    PcuNumericalMode::Strict
                };
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
            }
        }
    }
    let mut portable = kernel;
    portable
        .numerical_requirements
        .numerical_options
        .reproducibility = PcuReproducibility::PortableV1;
    assert!(PcuCpuIdentity.prepare_host_kernel(&portable).is_err());
    let mut plan = PcuCpuIdentity.prepare_host_kernel(&kernel).unwrap();
    let input = T::pattern(71);
    let sentinel = T::pattern(17);
    let mut output = [sentinel; 10];
    // Every invalid role table is rejected before the first byte is published.
    assert!(
        plan.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), core::slice::from_ref(&input)),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &output)
        ])
        .is_err()
    );
    assert!(
        plan.call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), core::slice::from_ref(&input))
        ])
        .is_err()
    );
    assert!(
        plan.call(&mut [PcuHostArgument::read(
            PcuBindingRef::new(0, 0),
            core::slice::from_ref(&input)
        )])
        .is_err()
    );
    super::same(&output, &[sentinel; 10]);
    plan.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), core::slice::from_ref(&input)),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
    ])
    .unwrap();
    super::same(&output[..7], &[input; 7]);
    super::same(&output[7..], &[sentinel; 3]);
}

//! Uses the independent role IR fixture, with the requested tuple retained cold.
#[path = "../../../../tests/checked_div_rem/roles/graph/graph.rs"]
mod base;
pub fn fixture<T: pcu_facade::PcuScalar, R>(
    count: u32,
    profile: usize,
    visit: impl FnOnce(&pcu_facade::PcuDispatchKernelIr<'_>) -> R,
) -> R {
    base::roles::fixture::<T, _>(count, profile, |original| {
        #[cfg(feature = "portable-joint-control")]
        let ir = pcu_facade::PcuDispatchKernelIr {
            numerical_requirements: pcu_facade::PcuImplementationRequirements {
                numerical_options: pcu_facade::PcuNumericalOptions {
                    reproducibility: pcu_facade::PcuReproducibility::PortableV1,
                    ..original.numerical_requirements.numerical_options
                },
                ..original.numerical_requirements
            },
            ..*original
        };
        #[cfg(not(feature = "portable-joint-control"))]
        let ir = *original;
        visit(&ir)
    })
}

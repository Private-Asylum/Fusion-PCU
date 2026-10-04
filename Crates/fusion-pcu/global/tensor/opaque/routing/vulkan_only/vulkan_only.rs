//! Sole Vulkan routing must work without an existing resident owner or native discovery.
use super::*;

#[test]
fn automatic_sole_vulkan_host_and_no_argument_routes_are_eligible() {
    let values = [1_u32, 2, 3];
    let inputs = [PcuTensorInput {
        kind: TensorInputKind::Host(&values),
        shape: PcuTensorShapeWitness::Dynamic(&[3]),
    }];
    assert!(selected_for_backend(PcuBackendChoice::Automatic, &inputs));
    assert!(selected_for_backend::<u32, 0>(
        PcuBackendChoice::Automatic,
        &[]
    ));
    assert!(automatic_vulkan_only(PcuBackendChoice::Automatic));
    assert!(selected_for_backend::<u32, 0>(
        PcuBackendChoice::Vulkan,
        &[]
    ));
    assert!(!automatic_vulkan_only(PcuBackendChoice::Vulkan));
    assert!(!automatic_vulkan_only(PcuBackendChoice::Rocm));
}

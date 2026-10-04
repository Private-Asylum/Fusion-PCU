//! Pure routing regressions avoid mutating global policy or discovering devices.
use super::*;
#[test]
fn explicit_device_route_defers_foreign_owner_validation_to_selected_inputs() {
    let owner = PcuTensor {
        backing: TensorBacking::Cpu {
            values: vec![1_u32, 2, 3],
            shape: std::rc::Rc::from([3]),
        },
    };
    let input = PcuTensorInput {
        kind: TensorInputKind::Resident(&owner),
        shape: PcuTensorShapeWitness::Dynamic(owner.shape()),
    };
    assert!(selected_for_backend(PcuBackendChoice::Automatic, &[input]));
    assert!(selected_for_backend(PcuBackendChoice::Cpu, &[input]));
    #[cfg(feature = "cuda")]
    assert!(!selected_for_backend(PcuBackendChoice::Cuda, &[input]));
    #[cfg(feature = "rocm")]
    assert!(!selected_for_backend(PcuBackendChoice::Rocm, &[input]));
}

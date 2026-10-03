#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuKernelId,
    PcuTypedDispatchValidationError,
    PcuValueTypeCaps,
};
use crate::PcuVulkanError;

fn fixture(run: impl FnOnce(PcuDispatchKernelIr<'_>)) {
    let bindings = [
        PcuBinding::scalar::<f32>(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<f32>(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
        ),
    ];
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: fusion_pcu::PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: fusion_pcu::PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    run(PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "offer",
            logical_shape: [17, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::FLOAT32,
        feature_caps: PcuDispatchFeatureCaps::default(),
    });
}

#[test]
fn recognized_scalar_faults_are_errors_not_absent_offers() {
    fixture(|kernel| {
        assert!(super::validate(&kernel).unwrap());
        let mut ops = kernel.ops.to_vec();
        if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { value, .. }) = &mut ops[1] {
            *value = PcuDispatchValueId(2);
        }
        let kernel = PcuDispatchKernelIr {
            ops: &ops,
            ..kernel
        };
        assert!(matches!(
            super::validate(&kernel),
            Err(PcuVulkanError::InvalidOfferValueFlow(
                PcuTypedDispatchValidationError::UndefinedValue(PcuDispatchValueId(2))
            ))
        ));
    });
    fixture(|kernel| {
        let mut bindings = kernel.bindings.to_vec();
        bindings[1] = bindings[0];
        let kernel = PcuDispatchKernelIr {
            bindings: &bindings,
            ..kernel
        };
        assert!(matches!(
            super::validate(&kernel),
            Err(PcuVulkanError::InvalidOfferBinding(_))
        ));
    });
    fixture(|mut kernel| {
        kernel.entry.logical_shape[0] = 0;
        assert!(matches!(
            super::validate(&kernel),
            Err(PcuVulkanError::InvalidOfferShape)
        ));
    });
    fixture(|kernel| {
        let mut bindings = kernel.bindings.to_vec();
        bindings[1].access = PcuBindingAccess::ReadOnly;
        let kernel = PcuDispatchKernelIr {
            bindings: &bindings,
            ..kernel
        };
        assert!(matches!(
            super::validate(&kernel),
            Err(PcuVulkanError::InvalidOfferBinding(PcuBindingRef {
                binding: 1,
                ..
            }))
        ));
    });
}

#[test]
fn valid_scalar_work_can_still_be_outside_the_vulkan_bit_map_profile() {
    fixture(|mut kernel| {
        // A valid declared two-dimensional source is outside this one-dimensional map.
        kernel.entry.logical_shape[1] = 2;
        assert!(super::validate(&kernel).unwrap());
        assert!(fusion_pcu_spirv::validate_float_bit_map(&kernel).is_err());
    });
}

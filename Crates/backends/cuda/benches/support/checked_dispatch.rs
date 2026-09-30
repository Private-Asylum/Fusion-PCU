//! Hardware fixture shared by the native-driver and PCU benchmark routes.
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaDiscovery,
    CudaOwnedDispatchBackend,
    CudaPreparedDispatch,
};
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuInvocationShape,
    PcuParameter,
    PcuPort,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuTargetDescriptor,
    PcuDeviceDescriptor,
    PcuObjectRef,
    PcuObjectKind,
    PcuDeviceClass,
    PcuRuntimeDiscovery,
};
use std::num::NonZeroU32;

pub fn selected_device() -> (CudaDiscovery, CudaOwnedDispatchBackend) {
    let discovery = CudaDiscovery::new();
    let invalid = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Device,
        id: 0,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    assert_eq!(discovery.providers(&mut providers).unwrap(), 1);
    let mut targets = [PcuTargetDescriptor {
        reference: invalid,
        name: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    assert_eq!(
        discovery
            .targets(providers[0].id, providers[0].generation, &mut targets)
            .unwrap(),
        1
    );
    let count = discovery.devices(targets[0].reference, &mut []).unwrap();
    assert!(count > 0, "test requires a visible CUDA device");
    let mut devices = vec![
        PcuDeviceDescriptor {
            reference: invalid,
            target: invalid,
            name: "",
            class: PcuDeviceClass::Other,
            vendor: None,
            architecture: None,
            generation: None,
            location: None,
        };
        count
    ];
    discovery
        .devices(targets[0].reference, &mut devices)
        .unwrap();
    let session = CudaOwnedDispatchBackend::open(&discovery, devices[0].reference, 2)
        .expect("open selected CUDA device");
    (discovery, session)
}

pub fn prepare(
    session: &CudaOwnedDispatchBackend,
    operation: PcuDispatchIntegerBinaryOp,
    extent: u32,
    grid_stride: bool,
) -> CudaPreparedDispatch {
    const LEFT: PcuBindingRef = PcuBindingRef::new(0, 0);
    const RIGHT: PcuBindingRef = PcuBindingRef::new(0, 1);
    const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 2);
    const LEFT_VALUE: PcuDispatchValueId = PcuDispatchValueId(1);
    const RIGHT_VALUE: PcuDispatchValueId = PcuDispatchValueId(2);
    const RESULT: PcuDispatchValueId = PcuDispatchValueId(3);
    let scalar_type = PcuValueType::Scalar(PcuScalarType::U32);
    let bindings = [
        PcuBinding::value(
            Some("left"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            scalar_type,
        ),
        PcuBinding::value(
            Some("right"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            scalar_type,
        ),
        PcuBinding::value(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            scalar_type,
        ),
    ];
    let index = if grid_stride {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: LEFT_VALUE,
            binding: LEFT,
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: RIGHT_VALUE,
            binding: RIGHT,
            index: PcuDispatchIndex::BindingElementZero,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: scalar_type,
            op: operation,
            result: RESULT,
            lhs: LEFT_VALUE,
            rhs: RIGHT_VALUE,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: OUTPUT,
            index,
            value: RESULT,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let grid_loop = [
        PcuDispatchOp::GridStrideLoop {
            extent,
            body: &body[..4],
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let ops = if grid_stride {
        &grid_loop[..]
    } else {
        &body[..]
    };
    let kernel = PcuDispatchKernelIr {
        id: fusion_pcu_core::PcuKernelId(91),
        entry: PcuDispatchEntryPoint {
            name: "checked_integer_map",
            logical_shape: [if grid_stride { 2 } else { extent }, 1, 1],
        },
        bindings: &bindings,
        ports: &[] as &[PcuPort<'_>],
        parameters: &[] as &[PcuParameter<'_>],
        ops,
        type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::U32),
        feature_caps: PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES),
    };
    let invocations =
        NonZeroU32::new(if grid_stride { 2 } else { extent }).expect("positive test extent");
    session
        .prepare_dispatch(PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(invocations),
        })
        .expect("prepare checked integer kernel")
}

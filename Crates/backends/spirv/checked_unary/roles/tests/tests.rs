//! Actual role projection cannot change the executable numerical module or admit Portable implicitly.
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
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use crate::{lower_checked_float_unary_to_spirv, PcuSpirvLoweringOptions};
#[rustfmt::skip]
use super::super::{lower_checked_float_unary_roles_to_spirv, validate_checked_float_unary_roles_map};
use std::vec::Vec;
fn module(
    scalar: PcuScalarType,
    operation: PcuDispatchFloatUnaryOp,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    grid: bool,
    broadcast: bool,
    roles: bool,
) -> Vec<u32> {
    let ty = PcuValueType::Scalar(scalar);
    let bindings = (0..if roles { 64 } else { 2 })
        .map(|index| {
            PcuBinding::value(
                None,
                0,
                index,
                PcuBindingStorageClass::Storage,
                if index == 1 {
                    PcuBindingAccess::ReadWrite
                } else {
                    PcuBindingAccess::ReadOnly
                },
                ty,
            )
        })
        .collect::<Vec<_>>();
    let input = if roles { 2 } else { 0 };
    let index = if grid {
        PcuDispatchIndex::GridStrideId
    } else {
        PcuDispatchIndex::InvocationId
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(7),
            binding: PcuBindingRef::new(0, input),
            index: if broadcast {
                PcuDispatchIndex::BindingElementZero
            } else {
                index
            },
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            result: PcuDispatchValueId(13),
            op: operation,
            value_type: ty,
            value: PcuDispatchValueId(7),
            underflow_policy: underflow,
            range_policy: range,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index,
            value: PcuDispatchValueId(13),
        }),
    ];
    let direct = [
        body[0],
        body[1],
        body[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let loop_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 7,
            body: &body,
        },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut kernel = PcuDispatchKernelIr {
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "unary_roles",
            logical_shape: [if grid { 3 } else { 7 }, 1, 1],
        },
        numerical_requirements: fusion_pcu::PcuImplementationRequirements {
            range_policy: range,
            float_underflow: underflow,
            ..Default::default()
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: if grid { &loop_ops } else { &direct },
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    };
    lower(&mut kernel, roles, broadcast, range)
}
fn lower(
    kernel: &mut PcuDispatchKernelIr<'_>,
    roles: bool,
    broadcast: bool,
    range: PcuRangePolicy,
) -> Vec<u32> {
    let mut words = Vec::new();
    if roles {
        let (_, profile) = lower_checked_float_unary_roles_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .unwrap();
        assert_eq!(profile.description.input_binding, PcuBindingRef::new(0, 2));
        assert_eq!(profile.description.output_binding, PcuBindingRef::new(0, 1));
        assert_eq!(
            profile.description.input_extent(),
            if broadcast { 1 } else { 7 }
        );
        assert_eq!(
            profile.description.requirements,
            kernel.numerical_requirements
        );
        assert!((18176..=18199).contains(&profile.local_id().unwrap()));
        kernel
            .numerical_requirements
            .numerical_options
            .reproducibility = PcuReproducibility::PortableV1;
        assert!(validate_checked_float_unary_roles_map(kernel).is_err());

        kernel
            .numerical_requirements
            .numerical_options
            .reproducibility = PcuReproducibility::Unspecified;
        kernel.numerical_requirements.range_policy = if range == PcuRangePolicy::Reject {
            PcuRangePolicy::Clamp
        } else {
            PcuRangePolicy::Reject
        };
    } else {
        lower_checked_float_unary_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .unwrap();
    }
    assert!(validate_checked_float_unary_roles_map(kernel).is_err());
    words
}
#[test]
fn all_six_actual_role_modules_equal_canonical_bytes_and_reject_header_changes() {
    let mut count = 0;
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        for operation in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
            for underflow in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            ] {
                for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    for grid in [false, true] {
                        for broadcast in [false, true] {
                            let actual =
                                module(scalar, operation, underflow, range, grid, broadcast, true);
                            let canonical =
                                module(scalar, operation, underflow, range, grid, broadcast, false);
                            assert_eq!(actual, canonical);
                            if let Ok(folder) = std::env::var("PCU_SPIRV_UNARY_ROLE_VALIDATION_DIR")
                            {
                                let bytes = actual
                                    .iter()
                                    .flat_map(|word| word.to_le_bytes())
                                    .collect::<Vec<_>>();
                                std::fs::write(
                                    std::path::Path::new(&folder)
                                        .join(std::format!("role-{count}.spv")),
                                    bytes,
                                )
                                .unwrap();
                            }
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(count, 288);
}

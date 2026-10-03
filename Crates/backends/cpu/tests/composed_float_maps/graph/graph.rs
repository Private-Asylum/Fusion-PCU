//! Hand-authored three-load/two-operation checked IR, independent from macro lowering.
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCheckedFloat,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuNumericalMode,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
};
pub fn prepare<T: PcuCheckedFloat, const N: usize>(
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> fusion_pcu_cpu::PcuCpuPreparedHost {
    let value_type = PcuValueType::Scalar(T::TYPE);
    let bindings = [
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
            value_type,
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            value_type,
        ),
    ];
    let load = |result| {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(result),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
        })
    };
    let binary = |result, lhs, rhs, op| {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            result: PcuDispatchValueId(result),
            lhs: PcuDispatchValueId(lhs),
            rhs: PcuDispatchValueId(rhs),
            op,
            value_type,
            range_policy: range,
            underflow_policy: underflow,
        })
    };
    let ops = [
        load(7),
        load(13),
        binary(17, 7, 13, PcuDispatchFloatBinaryOp::Add),
        load(29),
        binary(31, 17, 29, PcuDispatchFloatBinaryOp::Mul),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 0),
            value: PcuDispatchValueId(31),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    fusion_pcu_cpu::PcuCpuHostBackend::scalar()
        .prepare_host_kernel(&PcuDispatchKernelIr {
            numerical_requirements: PcuImplementationRequirements {
                numerical_mode: if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                    PcuNumericalMode::Strict
                } else {
                    PcuNumericalMode::Boundary
                },
                float_underflow: underflow,
                range_policy: range,
                ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
            },
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "composed_independent_graph",
                logical_shape: [u32::try_from(N).unwrap(), 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        })
        .unwrap()
}

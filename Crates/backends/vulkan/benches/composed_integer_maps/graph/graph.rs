//! Hand-authored three-load/two-operation checked IR, independent from macro lowering.
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCheckedInteger,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuValueType,
    PcuValueTypeCaps,
};
pub fn prepare<T: PcuCheckedInteger, const N: usize, B: PcuHostKernelBackend, const ONE: bool>(
    backend: &B,
    requirements: PcuImplementationRequirements,
) -> B::Prepared
where
    B::Error: core::fmt::Debug,
{
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
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            result: PcuDispatchValueId(result),
            lhs: PcuDispatchValueId(lhs),
            rhs: PcuDispatchValueId(rhs),
            op,
            value_type,
            range_policy: requirements.range_policy,
        })
    };
    let ops = [
        load(7),
        load(13),
        binary(17, 7, 13, PcuDispatchIntegerBinaryOp::Add),
        load(29),
        binary(31, 17, 29, PcuDispatchIntegerBinaryOp::Mul),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 0),
            value: PcuDispatchValueId(31),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let dead = [
        load(7),
        load(13),
        binary(17, 7, 13, PcuDispatchIntegerBinaryOp::Add),
        load(29),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 0),
            value: PcuDispatchValueId(29),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    backend
        .prepare_host_kernel(&PcuDispatchKernelIr {
            numerical_requirements: requirements,
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "composed_independent_graph",
                logical_shape: [u32::try_from(N).unwrap(), 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if ONE { &dead } else { &ops },
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        })
        .unwrap()
}

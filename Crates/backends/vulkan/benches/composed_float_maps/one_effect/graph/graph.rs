//! Independent one-effect SSA: load, checked unused Add, fresh load, store.
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
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuValueType,
    PcuValueTypeCaps,
};
pub fn prepare<T: PcuCheckedFloat, const N: usize, B: PcuHostKernelBackend>(
    backend: &B,
    requirements: PcuImplementationRequirements,
) -> B::Prepared
where
    B::Error: core::fmt::Debug,
{
    let ty = PcuValueType::Scalar(T::TYPE);
    let bindings = [
        PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            ty,
        ),
        PcuBinding::value(
            None,
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
            ty,
        ),
    ];
    let load = |result| {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(result),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        })
    };
    let ops = [
        load(7),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            result: PcuDispatchValueId(13),
            lhs: PcuDispatchValueId(7),
            rhs: PcuDispatchValueId(7),
            op: PcuDispatchFloatBinaryOp::Add,
            value_type: ty,
            range_policy: requirements.range_policy,
            underflow_policy: requirements.float_underflow,
        }),
        load(29),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            value: PcuDispatchValueId(29),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        numerical_requirements: requirements,
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "independent_one_effect",
            logical_shape: [u32::try_from(N).unwrap(), 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    };
    super::assert_policy(&kernel, requirements);
    backend.prepare_host_kernel(&kernel).unwrap()
}

//! Explicit diagnostic IR uses precisely the source entry's checked unary semantics.
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
    PcuValueTypeCaps,
};
use super::oracle::Scalar;

pub fn with_ir<T: Scalar, const N: usize, R>(
    call: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            T::VALUE_TYPE,
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            T::VALUE_TYPE,
        ),
    ];
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: T::VALUE_TYPE,
            op: PcuDispatchFloatUnaryOp::Neg,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    call(&PcuDispatchKernelIr {
        id: PcuKernelId(
            0xc0de_0000 ^ u32::try_from(N).unwrap() ^ u32::try_from(size_of::<T>()).unwrap(),
        ),
        entry: PcuDispatchEntryPoint {
            name: "cuda_checked_neg",
            logical_shape: [u32::try_from(N).unwrap(), 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: T::CAPS.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    })
}

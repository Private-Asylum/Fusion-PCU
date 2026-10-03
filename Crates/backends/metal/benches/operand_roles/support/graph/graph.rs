//! Handwritten IR has the same original declarations, exact opcode and one actual resource.
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuKernelId,
    PcuScalar,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
};
use super::Kind;
pub fn fixture<T: PcuScalar, R>(
    count: usize,
    kind: Kind,
    range: PcuRangePolicy,
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let bindings = [
        PcuBinding::scalar::<T>(
            Some("unread"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
        ),
        PcuBinding::scalar::<T>(
            Some("input"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
    ];
    let index = PcuDispatchIndex::InvocationId;
    let binary = match kind {
        Kind::Integer(op) => PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: PcuValueType::Scalar(T::TYPE),
            op,
            range_policy: range,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        },
        Kind::Float(op, underflow_policy) => PcuDispatchDataOp::CheckedFloatBinary {
            value_type: PcuValueType::Scalar(T::TYPE),
            op,
            range_policy: range,
            underflow_policy,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        },
    };
    let body = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: bindings[2].reference(),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: bindings[2].reference(),
            index,
        }),
        PcuDispatchOp::Data(binary),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: bindings[1].reference(),
            index,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut requirements = PcuDispatchKernelIr::DEFAULT_REQUIREMENTS;
    requirements.range_policy = range;
    if let Kind::Float(_, underflow) = kind {
        requirements.float_underflow = underflow;
    }
    visit(&PcuDispatchKernelIr {
        id: PcuKernelId(729),
        entry: PcuDispatchEntryPoint {
            name: "repeated-operand-roles",
            logical_shape: [u32::try_from(count).unwrap(), 1, 1],
        },
        numerical_requirements: requirements,
        bindings: &bindings,
        ops: &body,
        ports: &[],
        parameters: &[],
        type_caps: PcuValueTypeCaps::for_scalar(T::TYPE),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}

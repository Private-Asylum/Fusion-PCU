//! Explicit unary graph diagnostic independent from source lowering.
#[rustfmt::skip]
use pcu_facade::{PcuBinding,PcuBindingAccess,PcuBindingRef,PcuBindingStorageClass,PcuDispatchControlOp,PcuDispatchDataOp,PcuDispatchEntryPoint,PcuDispatchFeatureCaps,PcuDispatchFloatUnaryOp,PcuDispatchIndex,PcuDispatchKernelIr,PcuDispatchOp,PcuDispatchValueId,PcuFloatUnderflowPolicy,PcuKernelId,PcuRangePolicy,PcuScalarType,PcuValueType,PcuValueTypeCaps};
#[derive(Clone, Copy)]
pub struct Graph {
    pub scalar: PcuScalarType,
    pub extent: u32,
    pub op: PcuDispatchFloatUnaryOp,
    pub policy: PcuFloatUnderflowPolicy,
    pub range: PcuRangePolicy,
    pub grid: bool,
    pub broadcast: bool,
}
impl Graph {
    pub const fn new(
        scalar: PcuScalarType,
        extent: u32,
        op: PcuDispatchFloatUnaryOp,
        policy: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
    ) -> Self {
        Self {
            scalar,
            extent,
            op,
            policy,
            range,
            grid: false,
            broadcast: false,
        }
    }
    pub fn with<R>(self, run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R) -> R {
        let ty = PcuValueType::Scalar(self.scalar);
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
                2,
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
            PcuBinding::value(
                None,
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                ty,
            ),
        ];
        let index = if self.grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(7),
                binding: PcuBindingRef::new(0, 2),
                index: if self.broadcast {
                    PcuDispatchIndex::BindingElementZero
                } else {
                    index
                },
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                result: PcuDispatchValueId(13),
                op: self.op,
                value_type: ty,
                value: PcuDispatchValueId(7),
                underflow_policy: self.policy,
                range_policy: self.range,
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
        let grid = [
            PcuDispatchOp::GridStrideLoop {
                extent: self.extent,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        run(&PcuDispatchKernelIr {
            numerical_requirements: pcu_facade::PcuImplementationRequirements {
                float_underflow: self.policy,
                range_policy: self.range,
                ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
            },
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "low_unary",
                logical_shape: [if self.grid { 3 } else { self.extent }, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if self.grid { &grid } else { &direct },
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        })
    }
}

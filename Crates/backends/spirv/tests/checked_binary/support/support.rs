//! Independent explicit graph diagnostic, shared by owned SPIR-V/Vulkan qualification.
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
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuKernelId,
    PcuRangePolicy,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[derive(Clone, Copy)]
pub struct Graph {
    pub extent: u32,
    pub op: PcuDispatchFloatBinaryOp,
    pub policy: PcuFloatUnderflowPolicy,
    pub range: PcuRangePolicy,
    pub scalar: PcuScalarType,
    pub grid: bool,
    pub reverse_loads: bool,
    pub operands: [u16; 2],
    pub broadcast: [bool; 2],
}
impl Graph {
    pub const fn new(
        extent: u32,
        op: PcuDispatchFloatBinaryOp,
        policy: PcuFloatUnderflowPolicy,
    ) -> Self {
        Self {
            extent,
            op,
            policy,
            range: PcuRangePolicy::Reject,
            scalar: PcuScalarType::F32,
            grid: false,
            reverse_loads: false,
            operands: [1, 2],
            broadcast: [false; 2],
        }
    }
    pub fn with<R>(self, run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R) -> R {
        let bindings = core::array::from_fn::<_, 3, _>(|index| {
            PcuBinding::value(
                None,
                0,
                u32::try_from(index).unwrap(),
                PcuBindingStorageClass::Storage,
                if index == 2 {
                    PcuBindingAccess::ReadWrite
                } else {
                    PcuBindingAccess::ReadOnly
                },
                PcuValueType::Scalar(self.scalar),
            )
        });
        let index = if self.grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let load = |bank: usize| {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(u16::try_from(bank + 1).unwrap()),
                binding: PcuBindingRef::new(0, u32::try_from(bank).unwrap()),
                index: if self.broadcast[bank] {
                    PcuDispatchIndex::BindingElementZero
                } else {
                    index
                },
            })
        };
        let mut body = [
            load(0),
            load(1),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                result: PcuDispatchValueId(3),
                value_type: PcuValueType::Scalar(self.scalar),
                op: self.op,
                range_policy: self.range,
                underflow_policy: self.policy,
                lhs: PcuDispatchValueId(self.operands[0]),
                rhs: PcuDispatchValueId(self.operands[1]),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index,
                value: PcuDispatchValueId(3),
            }),
        ];
        if self.reverse_loads {
            body.swap(0, 1);
        }
        let direct = [
            body[0],
            body[1],
            body[2],
            body[3],
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
            numerical_requirements: fusion_pcu_core::PcuImplementationRequirements {
                range_policy: self.range,
                float_underflow: self.policy,
                ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
            },
            id: PcuKernelId(71),
            entry: PcuDispatchEntryPoint {
                name: "binary_diagnostic",
                logical_shape: [if self.grid { 2 } else { self.extent }, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if self.grid { &grid } else { &direct },
            type_caps: PcuValueTypeCaps::for_scalar(self.scalar),
            feature_caps: PcuDispatchFeatureCaps::default(),
        })
    }
}

//! Independent typed integer diagnostic with deliberately reordered noncanonical binding roles.
#[rustfmt::skip]
use fusion_pcu_core::{PcuBinding,PcuBindingAccess,PcuBindingRef,PcuBindingStorageClass,PcuDispatchControlOp,PcuDispatchDataOp,PcuDispatchEntryPoint,PcuDispatchFeatureCaps,PcuDispatchIntegerBinaryOp,PcuDispatchIndex,PcuDispatchKernelIr,PcuDispatchOp,PcuDispatchValueId,PcuKernelId,PcuRangePolicy,PcuScalarType,PcuValueType,PcuValueTypeCaps};
pub const INPUTS: [PcuBindingRef; 2] = [PcuBindingRef::new(7, 9), PcuBindingRef::new(9, 3)];
pub const OUTPUT: PcuBindingRef = PcuBindingRef::new(8, 11);
#[derive(Clone, Copy)]
pub struct Graph {
    pub scalar: PcuScalarType,
    pub extent: u32,
    pub op: PcuDispatchIntegerBinaryOp,
    pub range: PcuRangePolicy,
    pub grid: bool,
    pub broadcast: bool,
    pub swapped: bool,
}
impl Graph {
    pub const fn new(
        scalar: PcuScalarType,
        extent: u32,
        op: PcuDispatchIntegerBinaryOp,
        range: PcuRangePolicy,
    ) -> Self {
        Self {
            scalar,
            extent,
            op,
            range,
            grid: false,
            broadcast: false,
            swapped: false,
        }
    }
    pub fn with<R>(self, run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R) -> R {
        let ty = PcuValueType::Scalar(self.scalar);
        let binding = |reference: PcuBindingRef, access| {
            PcuBinding::value(
                None,
                reference.set,
                reference.binding,
                PcuBindingStorageClass::Storage,
                access,
                ty,
            )
        };
        let bindings = [
            binding(OUTPUT, PcuBindingAccess::WriteOnly),
            binding(INPUTS[0], PcuBindingAccess::ReadOnly),
            binding(INPUTS[1], PcuBindingAccess::ReadOnly),
        ];
        let index = if self.grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(7),
                binding: INPUTS[1],
                index: if self.broadcast {
                    PcuDispatchIndex::BindingElementZero
                } else {
                    index
                },
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(11),
                binding: INPUTS[0],
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                result: PcuDispatchValueId(13),
                op: self.op,
                value_type: ty,
                range_policy: self.range,
                lhs: PcuDispatchValueId(if self.swapped { 7 } else { 11 }),
                rhs: PcuDispatchValueId(if self.swapped { 11 } else { 7 }),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: OUTPUT,
                index,
                value: PcuDispatchValueId(13),
            }),
        ];
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
                ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
            },
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "checked_integer",
                logical_shape: [if self.grid { 3 } else { self.extent }, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if self.grid { &grid } else { &direct },
            type_caps: PcuValueTypeCaps::for_scalar(self.scalar),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        })
    }
}

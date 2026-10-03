//! Independent five-operation diagnostic with noncanonical bindings and SSA routing.
#[rustfmt::skip]
use fusion_pcu_core::{PcuBinding,PcuBindingAccess,PcuBindingRef,PcuBindingStorageClass,PcuDispatchControlOp,PcuDispatchDataOp,PcuDispatchEntryPoint,PcuDispatchFeatureCaps,PcuDispatchIndex,PcuDispatchKernelIr,PcuDispatchOp,PcuDispatchValueId,PcuKernelId,PcuScalarType,PcuValueType,PcuValueTypeCaps};
use fusion_pcu_core::model::PcuIntegerDivFlags;
pub const INPUTS: [PcuBindingRef; 2] = [PcuBindingRef::new(7, 9), PcuBindingRef::new(9, 3)];
pub const OUTPUTS: [PcuBindingRef; 2] = [PcuBindingRef::new(8, 11), PcuBindingRef::new(2, 8)];
#[derive(Clone, Copy)]
pub struct Graph {
    pub scalar: PcuScalarType,
    pub extent: u32,
    pub grid: bool,
    pub kind: usize,
}
impl Graph {
    pub const fn new(scalar: PcuScalarType, extent: u32) -> Self {
        Self {
            scalar,
            extent,
            grid: false,
            kind: 0,
        }
    }
    pub fn with<R>(self, run: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R) -> R {
        let ty = PcuValueType::Scalar(self.scalar);
        let binding = |r: PcuBindingRef, access| {
            PcuBinding::value(
                None,
                r.set,
                r.binding,
                PcuBindingStorageClass::Storage,
                access,
                ty,
            )
        };
        let declarations = declarations(self.kind, binding);
        let bindings = if self.kind == 0 || self.kind == 3 {
            &declarations[..3]
        } else {
            &declarations[..]
        };
        let index = if self.grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(7),
                binding: INPUTS[0],
                index: if self.kind == 4 || self.kind == 5 {
                    PcuDispatchIndex::BindingElementZero
                } else {
                    index
                },
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(11),
                binding: if self.kind == 2 || self.kind == 5 {
                    INPUTS[1]
                } else {
                    INPUTS[0]
                },
                index: if self.kind == 3 || self.kind == 5 {
                    PcuDispatchIndex::BindingElementZero
                } else {
                    index
                },
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: ty,
                flags: PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(13),
                remainder: PcuDispatchValueId(15),
                lhs: PcuDispatchValueId(7),
                rhs: PcuDispatchValueId(11),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: OUTPUTS[0],
                index,
                value: PcuDispatchValueId(13),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: OUTPUTS[1],
                index,
                value: PcuDispatchValueId(15),
            }),
        ];
        let direct = [
            body[0],
            body[1],
            body[2],
            body[3],
            body[4],
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
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "checked_div_rem",
                logical_shape: [if self.grid { 3 } else { self.extent }, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops: if self.grid { &grid } else { &direct },
            type_caps: PcuValueTypeCaps::for_scalar(self.scalar),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        })
    }
}

fn declarations(
    kind: usize,
    binding: impl Fn(PcuBindingRef, PcuBindingAccess) -> PcuBinding<'static>,
) -> [PcuBinding<'static>; 4] {
    let q = binding(OUTPUTS[0], PcuBindingAccess::WriteOnly);
    let r = binding(OUTPUTS[1], PcuBindingAccess::WriteOnly);
    let a = binding(INPUTS[0], PcuBindingAccess::ReadOnly);
    let b = binding(INPUTS[1], PcuBindingAccess::ReadOnly);
    let unused = binding(PcuBindingRef::new(5, 17), PcuBindingAccess::ReadOnly);
    match kind {
        0 | 3 => [q, r, a, unused],
        1 => [unused, a, r, q],
        2 => [b, r, a, q],
        4 => [r, unused, q, a],
        _ => [a, q, b, r],
    }
}

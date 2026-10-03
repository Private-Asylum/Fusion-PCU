//! Independent cold role mutations of explicit quotient/remainder SSA, never hot source lowering.
#[rustfmt::skip]
use pcu_facade::{
    PcuScalar,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchControlOp,
    PcuDispatchOp,
    PcuDispatchKernelIr,
};
// The canonical benchmark includes the shared parent fixture without exercising its role extension.
#[allow(dead_code)]
pub fn fixture<T: PcuScalar, R>(
    count: u32,
    profile: usize,
    visit: impl FnOnce(&PcuDispatchKernelIr<'_>) -> R,
) -> R {
    let grid = matches!(profile, 3 | 5);
    super::fixture::<T, _>(
        count,
        grid,
        false,
        pcu_facade::PcuImplementationRequirements {
            numerical_mode: if profile == 3 {
                pcu_facade::PcuNumericalMode::Strict
            } else {
                pcu_facade::PcuNumericalMode::Boundary
            },
            ..pcu_facade::PcuImplementationRequirements::default()
        },
        |original| {
            let baseline = match original.ops[0] {
                PcuDispatchOp::GridStrideLoop { body, .. } => body,
                _ => &original.ops[..5],
            };
            let mut body = baseline.to_vec();
            if matches!(profile, 0 | 1 | 3 | 5) {
                let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, .. }) =
                    &mut body[1]
                else {
                    unreachable!()
                };
                *binding = original.bindings[0].reference();
            }
            for (slot, broadcast) in [matches!(profile, 3 | 5), matches!(profile, 1 | 4 | 5)]
                .into_iter()
                .enumerate()
            {
                if broadcast {
                    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { index, .. }) =
                        &mut body[slot]
                    else {
                        unreachable!()
                    };
                    *index = PcuDispatchIndex::BindingElementZero;
                }
            }
            let direct = [
                body[0],
                body[1],
                body[2],
                body[3],
                body[4],
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let looped = [
                PcuDispatchOp::GridStrideLoop {
                    extent: count,
                    body: &body,
                },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let mut entry = original.entry;
            if grid {
                entry.logical_shape[0] = 3;
            }
            visit(&PcuDispatchKernelIr {
                entry,
                ops: if grid { &looped } else { &direct },
                ..*original
            })
        },
    )
}

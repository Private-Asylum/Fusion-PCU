//! Exact role offers and truthful actual-input session dispatch.
#[rustfmt::skip]
use super::super::{
    same,
    Sample,
};
use super::super::graph::roles as graph;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxDiscovery,
    MlxCheckedDivRemRolePlan,
    MlxPreparedDispatchKernel,
    MlxDispatchOutputLayout,
    MLX_EXECUTOR,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCostBoundary,
    PcuDeviceActivation,
    PcuDeviceIdentity,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationCost,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuPreparedHostKernel,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};
fn qualify<T: Sample>(discovery: &MlxDiscovery) {
    let reference = discovery.device_reference(0).unwrap();
    let device = PcuDeviceIdentity::from_device_ref(reference).unwrap();
    let session = discovery.open_device(reference).unwrap();
    for profile in [0, 1, 3, 4, 5] {
        graph::fixture::<T, _>(5, profile, |ir| {
            let plan = MlxCheckedDivRemRolePlan::assess(ir).unwrap();
            let request = PcuImplementationRequest {
                device,
                executor: MLX_EXECUTOR,
                operation: ir,
                requirements: ir.numerical_requirements,
                boundary: PcuCostBoundary::Host,
            };
            let mut offers = [None];
            assert_eq!(
                discovery
                    .implementation_offers(&request, &mut offers)
                    .unwrap(),
                1
            );
            let offer = offers[0].unwrap();
            offer.validate_request(&request).unwrap();
            assert_eq!(
                offer.implementation.local_id,
                plan.implementation_local_id()
            );
            assert_eq!(offer.implementation.revision, 0x0003_0020_0003_0e00);
            assert_eq!(
                offer.cost,
                PcuImplementationCost::unknown(PcuCostBoundary::Host)
            );
            assert_eq!(offer.workspace_bytes, None);
            let mut prepared = session.prepare_host_kernel(ir).unwrap();
            assert!(matches!(
                prepared,
                MlxPreparedDispatchKernel::DivRemRoles(_)
            ));
            assert_eq!(prepared.input_bindings(), plan.input_bindings());
            assert_eq!(
                prepared.output_layout(),
                MlxDispatchOutputLayout::DivRem {
                    bindings: plan.output_bindings(),
                    byte_lengths: [plan.byte_len(); 2]
                }
            );
            let data = [T::minimum(), T::raw(2), T::raw(3), T::raw(4), T::raw(5)];
            let right = [T::raw(3); 5];
            let sentinel = T::raw(91);
            let mut q = [sentinel; 7];
            let mut r = [sentinel; 7];
            let inputs = plan.input_bindings();
            let outputs = plan.output_bindings();
            if inputs.len() == 1 {
                prepared
                    .call(&mut [
                        PcuHostArgument::read_write(outputs[1], &mut r),
                        PcuHostArgument::read(inputs[0], &data),
                        PcuHostArgument::read_write(outputs[0], &mut q),
                    ])
                    .unwrap();
            } else {
                prepared
                    .call(&mut [
                        PcuHostArgument::read_write(outputs[1], &mut r),
                        PcuHostArgument::read(inputs[0], &data),
                        PcuHostArgument::read_write(outputs[0], &mut q),
                        PcuHostArgument::read(inputs[1], &right),
                    ])
                    .unwrap();
            }
            let values = [&data, &right];
            let roles = plan.operand_inputs();
            let broadcast = plan.operand_broadcast();
            for lane in 0..5 {
                let a = values[roles[0]][if broadcast[0] { 0 } else { lane }];
                let b = values[roles[1]][if broadcast[1] { 0 } else { lane }];
                same(&q[lane..=lane], &[a.pcu_checked_div(b).unwrap()]);
                same(&r[lane..=lane], &[a.pcu_checked_rem(b).unwrap()]);
            }
            same(&q[5..], &[sentinel; 2]);
            same(&r[5..], &[sentinel; 2]);
            assert_eq!(
                discovery
                    .implementation_offers(
                        &PcuImplementationRequest {
                            boundary: PcuCostBoundary::Resident,
                            ..request
                        },
                        &mut offers
                    )
                    .unwrap(),
                0
            );
        });
    }
}
#[test]
#[ignore = "required native MLX actual-read division exact offer and session aggregate qualification"]
fn fourteen_width_role_offers_and_session_aggregate() {
    let discovery = MlxDiscovery::discover_default().unwrap();
    macro_rules! types {($($ty:ty),+) => {$(qualify::<$ty>(&discovery);)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

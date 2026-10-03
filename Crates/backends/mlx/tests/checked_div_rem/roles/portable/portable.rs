//! Requested Portable headers retain their complete cold tuple and actual input roles.
#[rustfmt::skip]
use super::super::{
    graph,
    Sample,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedDivRemPlan,
    MlxCheckedDivRemRolePlan,
};
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{
    PcuDeviceActivation,
    PcuDeviceIdentity,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuPreparedHostKernel,
    PcuCompoundArithmeticPolicy,
    PcuDispatchKernelIr,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};
fn qualify<T: Sample>() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for underflow in [
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    let requirements = PcuImplementationRequirements {
                        numerical_mode: mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            reproducibility: PcuReproducibility::PortableV1,
                        },
                        float_underflow: underflow,
                        ..Default::default()
                    };
                    for profile in 0..6 {
                        graph::roles::fixture::<T, _>(65, profile, |original| {
                            let ir = PcuDispatchKernelIr {
                                numerical_requirements: requirements,
                                ..*original
                            };
                            let descriptor =
                                pcu_facade::describe_portable_v1_integer_div_rem_map(&ir).unwrap();
                            let plan = MlxCheckedDivRemRolePlan::assess(&ir).unwrap();
                            assert_eq!(plan.scalar_type(), T::TYPE);
                            assert_eq!(plan.requirements(), requirements);
                            assert_eq!(plan.element_count(), 65);
                            assert_eq!(plan.input_bindings(), descriptor.operands.input_bindings());
                            assert_eq!(plan.operand_inputs(), descriptor.operands.operand_inputs());
                            assert_eq!(
                                plan.output_bindings(),
                                descriptor.operands.output_bindings()
                            );
                            let ordinary = MlxCheckedDivRemRolePlan::assess(original).unwrap();
                            assert_eq!(
                                plan.implementation_local_id(),
                                ordinary.implementation_local_id() + 0x1000
                            );
                            if profile == 2 {
                                let canonical = MlxCheckedDivRemPlan::assess(&ir).unwrap();
                                assert_eq!(canonical.requirements(), requirements);
                                assert_eq!(
                                    canonical.implementation_local_id(),
                                    ordinary.implementation_local_id() - 0x100 + 0x1000
                                );
                            }
                            let clamp = PcuDispatchKernelIr {
                                numerical_requirements: PcuImplementationRequirements {
                                    range_policy: PcuRangePolicy::Clamp,
                                    ..requirements
                                },
                                ..ir
                            };
                            assert!(MlxCheckedDivRemRolePlan::assess(&clamp).is_err());
                            let zero = PcuDispatchKernelIr {
                                entry: pcu_facade::PcuDispatchEntryPoint {
                                    logical_shape: [0, 1, 1],
                                    ..ir.entry
                                },
                                ..ir
                            };
                            assert!(MlxCheckedDivRemRolePlan::assess(&zero).is_err());
                        });
                    }
                }
            }
        }
    }
}
#[test]
fn fourteen_width_portable_joint_tuple_and_actual_roles() {
    macro_rules! types {($($ty:ty),+) => {$(qualify::<$ty>();)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

struct Detached;
impl PcuHostKernelBackend for Detached {
    type Prepared = Self;
    type Error = fusion_pcu_mlx::MlxHostKernelError;
    fn prepare_host_kernel(&self, kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, Self::Error> {
        let plan = MlxCheckedDivRemRolePlan::assess(kernel)
            .map_err(pcu_facade::PcuHostDispatchError::Backend)?;
        assert_eq!(
            plan.requirements().numerical_options.reproducibility,
            PcuReproducibility::PortableV1
        );
        Ok(Self)
    }
}
impl PcuPreparedHostKernel for Detached {
    type Error = fusion_pcu_mlx::MlxHostKernelError;
    fn call(&mut self, _: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        Err(pcu_facade::PcuHostDispatchError::Backend(
            fusion_pcu_mlx::MlxError::InvalidRequest("detached qualification only".into()),
        ))
    }
}
fn source_headers<T: Sample>() {
    assert!(source::repeated_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::unread_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::reordered_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::grid_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::scalar_divisor_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::scalar_grid_prepare::<T, 65, _>(&Detached).is_ok());
}
#[test]
fn all_fourteen_genuine_source_specializations_request_portable_before_runtime() {
    macro_rules! types {($($ty:ty),+) => {$(source_headers::<$ty>();)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

#[allow(clippy::too_many_lines)] // One profile joins exact offer/aggregate, both owner outputs and fatal sibling liveness assertions.
fn native<T: Sample>(discovery: &fusion_pcu_mlx::MlxDiscovery) {
    let reference = discovery.device_reference(0).unwrap();
    let device = PcuDeviceIdentity::from_device_ref(reference).unwrap();
    let session = discovery.open_device(reference).unwrap();
    let left = [T::raw(23), T::minimum(), T::raw(3), T::raw(4), T::raw(5)];
    let right = [T::raw(3); 5];
    let sentinel = T::raw(91);
    for profile in 0..6 {
        graph::roles::fixture::<T, _>(5, profile, |original| {
            let ir = PcuDispatchKernelIr {
                numerical_requirements: PcuImplementationRequirements {
                    numerical_mode: PcuNumericalMode::Strict,
                    float_underflow: PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    numerical_options: PcuNumericalOptions {
                        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
                        precision: PcuPrecisionPolicy::BackendOptimized,
                        reproducibility: PcuReproducibility::PortableV1,
                    },
                    ..Default::default()
                },
                ..*original
            };
            let plan = MlxCheckedDivRemRolePlan::assess(&ir).unwrap();
            let request = PcuImplementationRequest {
                device,
                executor: fusion_pcu_mlx::MLX_EXECUTOR,
                operation: &ir,
                requirements: ir.numerical_requirements,
                boundary: pcu_facade::PcuCostBoundary::Host,
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
            assert_eq!(offer.implementation.revision, 0x0003_0020_0003_1100);
            let expected = MlxCheckedDivRemPlan::assess(&ir).map_or_else(
                |_| plan.implementation_local_id(),
                |p| p.implementation_local_id(),
            );
            assert_eq!(offer.implementation.local_id, expected);
            let mut aggregate = session.prepare_host_kernel(&ir).unwrap();
            assert_eq!(aggregate.input_bindings(), plan.input_bindings());
            let values = [&left[..], &right[..]];
            let inputs = plan.input_bindings();
            let counts = plan.input_element_counts();
            let outputs = plan.output_bindings();
            let mut q = [sentinel; 7];
            let mut r = [sentinel; 9];
            let mut arguments = vec![
                PcuHostArgument::read_write(outputs[0], &mut q),
                PcuHostArgument::read_write(outputs[1], &mut r),
            ];
            for slot in 0..inputs.len() {
                arguments.push(PcuHostArgument::read(
                    inputs[slot],
                    &values[slot][..counts[slot]],
                ));
            }
            aggregate.call(&mut arguments).unwrap();
            drop(arguments);
            let roles = plan.operand_inputs();
            let broadcast = plan.operand_broadcast();
            for lane in 0..5 {
                let a = values[roles[0]][if broadcast[0] { 0 } else { lane }];
                let b = values[roles[1]][if broadcast[1] { 0 } else { lane }];
                super::super::same(&q[lane..=lane], &[a.pcu_checked_div(b).unwrap()]);
                super::super::same(&r[lane..=lane], &[a.pcu_checked_rem(b).unwrap()]);
            }
            super::super::same(&q[5..], &[sentinel; 2]);
            super::super::same(&r[5..], &[sentinel; 4]);
            let expected_q = q;
            let expected_r = r;
            let owners: Vec<_> = (0..inputs.len())
                .map(|slot| {
                    session
                        .upload_encoded(&values[slot][..counts[slot]])
                        .unwrap()
                })
                .collect();
            let mut role_kernel = session
                .checked_div_rem_role_backend()
                .prepare_host_kernel(&ir)
                .unwrap();
            let incoming: Vec<_> = inputs
                .iter()
                .zip(&owners)
                .map(|(&target, array)| fusion_pcu_mlx::MlxBinaryInput::Resident { target, array })
                .collect();
            let [qo, ro] = role_kernel.execute_inputs(&incoming).unwrap();
            qo.read_into(&mut q).unwrap();
            ro.read_into(&mut r).unwrap();
            super::super::same(&q, &expected_q);
            super::super::same(&r, &expected_r);
            let old_q = qo.clone();
            let zeros: Vec<_> = counts[..inputs.len()]
                .iter()
                .map(|&count| session.upload_encoded(&vec![T::raw(0); count]).unwrap())
                .collect();
            let bad: Vec<_> = inputs
                .iter()
                .zip(&zeros)
                .map(|(&target, array)| fusion_pcu_mlx::MlxBinaryInput::Resident { target, array })
                .collect();
            assert!(
                matches!(role_kernel.execute_inputs(&bad), Err(fusion_pcu_mlx::MlxError::Arithmetic(fault)) if fault.kind == pcu_facade::PcuExecutionFaultKind::DivideByZero && fault.invocation_id==0 && !fault.recovered)
            );
            assert!(!role_kernel.last_call_may_have_written_existing_encoded_owner());
            old_q.read_into(&mut q).unwrap();
            ro.read_into(&mut r).unwrap();
            super::super::same(&q, &expected_q);
            super::super::same(&r, &expected_r);
            super::super::same(&q[5..], &[sentinel; 2]);
            super::super::same(&r[5..], &[sentinel; 4]);
            old_q.release().unwrap();
            qo.release().unwrap();
            ro.release().unwrap();
            for owner in owners.into_iter().chain(zeros) {
                owner.release().unwrap();
            }
        });
    }
}
#[test]
#[ignore = "required native Portable fourteen-width exact offers, source metadata and resident joint faults"]
fn fourteen_width_portable_offers_and_private_joint_publication() {
    let discovery = fusion_pcu_mlx::MlxDiscovery::discover_default().unwrap();
    macro_rules! types {($($ty:ty),+) => {$(native::<$ty>(&discovery);)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

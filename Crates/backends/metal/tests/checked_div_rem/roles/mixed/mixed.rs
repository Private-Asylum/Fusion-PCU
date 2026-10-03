//! Actual unique resident leases, host minima and joint prefix publication over all14 types.
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalDivRemRolePlan,
    MetalError,
    MetalHostKernelError,
    MetalMixedHostArgument,
    MetalPreparedDivRemRoleHostKernel,
    MetalSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBindingRef,
    PcuDeviceArgument,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuMemoryPoolId,
    PcuReproducibility,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
};
#[rustfmt::skip]
use super::super::{
    mixed::{
        owner,
        read,
    },
    same,
    Sample,
};

struct Call<'a> {
    prepared: &'a mut MetalPreparedDivRemRoleHostKernel,
    session: &'a MetalSession,
    foreign: &'a MetalSession,
    plan: MetalDivRemRolePlan,
    unread: Option<PcuBindingRef>,
    mode: usize,
}
impl Call<'_> {
    fn run<T: Sample>(
        &mut self,
        a: &[T],
        b: &[T],
        q: &mut [T],
        r: &mut [T],
    ) -> Result<(), MetalHostKernelError> {
        let mut provider = self.session.memory_provider(PcuMemoryPoolId(121));
        let mut foreign = self.foreign.memory_provider(PcuMemoryPoolId(121));
        let values = [a, b];
        let inputs = [owner(&mut provider, a), owner(&mut provider, b)];
        let ignored = owner(&mut foreign, &[T::raw(0)]);
        let mut q_owner = owner(&mut provider, q);
        let mut r_owner = owner(&mut provider, r);
        let before = (q.to_vec(), r.to_vec());
        let targets = self.plan.output_bindings();
        let mut arguments = Vec::new();
        for (slot, &target) in self.plan.input_bindings().iter().enumerate() {
            if self.mode == 4 || (self.mode == 1 && slot == 0) || (self.mode == 2 && slot == 1) {
                let minimum = self.plan.input_element_counts()[slot];
                arguments.push(MetalMixedHostArgument::Host(PcuHostArgument::read(
                    target,
                    &values[slot][..minimum],
                )));
            } else {
                arguments.push(MetalMixedHostArgument::Resident(PcuDeviceArgument::read(
                    target,
                    &inputs[slot],
                )));
            }
        }
        if let Some(target) = self.unread {
            arguments.push(MetalMixedHostArgument::Resident(PcuDeviceArgument::read(
                target, &ignored,
            )));
        }
        if self.mode == 1 || self.mode >= 3 {
            arguments.push(MetalMixedHostArgument::Host(PcuHostArgument::read_write(
                targets[0], q,
            )));
        } else {
            arguments.push(MetalMixedHostArgument::Resident(
                PcuDeviceArgument::read_write(targets[0], &mut q_owner),
            ));
        }
        if self.mode == 2 || self.mode >= 3 {
            arguments.push(MetalMixedHostArgument::Host(PcuHostArgument::read_write(
                targets[1], r,
            )));
        } else {
            arguments.push(MetalMixedHostArgument::Resident(
                PcuDeviceArgument::read_write(targets[1], &mut r_owner),
            ));
        }
        let result = self.prepared.call_mixed(&mut arguments);
        drop(arguments);
        assert!(!self.prepared.last_call_completion_uncertain());
        if result.is_err() {
            assert!(!self.prepared.last_call_may_have_written());
            same(&read(&mut provider, &q_owner), &before.0);
            same(&read(&mut provider, &r_owner), &before.1);
        } else {
            assert!(self.prepared.last_call_may_have_written());
            if self.mode != 1 && self.mode < 3 {
                q.copy_from_slice(&read(&mut provider, &q_owner));
            }
            if self.mode != 2 && self.mode < 3 {
                r.copy_from_slice(&read(&mut provider, &r_owner));
            }
        }
        same(&read(&mut foreign, &ignored), &[T::raw(0)]);
        result
    }
}

fn qualify<T: Sample>(session: &MetalSession, foreign: &MetalSession) {
    for profile in 0..6 {
        for portable in [false, true] {
            let (mut prepared, plan, unread) =
                super::graph::roles::fixture::<T, _>(65, profile, |ir| {
                    let mut ir = *ir;
                    if portable {
                        ir.numerical_requirements.numerical_options.reproducibility =
                            PcuReproducibility::PortableV1;
                    }
                    let plan = MetalDivRemRolePlan::assess(&ir).unwrap();
                    let unread =
                        ir.bindings
                            .iter()
                            .map(|binding| binding.reference())
                            .find(|target| {
                                !plan.input_bindings().contains(target)
                                    && !plan.output_bindings().contains(target)
                            });
                    (
                        session
                            .checked_div_rem_role_backend()
                            .prepare_host_kernel(&ir)
                            .unwrap(),
                        plan,
                        unread,
                    )
                });
            for mode in 0..5 {
                let mut call = Call {
                    prepared: &mut prepared,
                    session,
                    foreign,
                    plan,
                    unread,
                    mode,
                };
                super::pairs::<T>(profile, |a, b, q, r| call.run(a, b, q, r));
            }
            foreign_input::<T>(session, foreign, &mut prepared, plan);
        }
    }
}
fn foreign_input<T: Sample>(
    session: &MetalSession,
    foreign: &MetalSession,
    prepared: &mut MetalPreparedDivRemRoleHostKernel,
    plan: MetalDivRemRolePlan,
) {
    let mut provider = session.memory_provider(PcuMemoryPoolId(121));
    let mut other = foreign.memory_provider(PcuMemoryPoolId(121));
    let foreign_input = owner(&mut other, &[T::raw(4); 65]);
    let good = owner(&mut provider, &[T::raw(4); 65]);
    let mut q = owner(&mut provider, &[T::raw(9); 68]);
    let mut r = owner(&mut provider, &[T::raw(9); 70]);
    let input = plan.input_bindings();
    let output = plan.output_bindings();
    let mut arguments = vec![MetalMixedHostArgument::Resident(PcuDeviceArgument::read(
        input[0],
        &foreign_input,
    ))];
    if input.len() == 2 {
        arguments.push(MetalMixedHostArgument::Resident(PcuDeviceArgument::read(
            input[1], &good,
        )));
    }
    arguments.push(MetalMixedHostArgument::Resident(
        PcuDeviceArgument::read_write(output[0], &mut q),
    ));
    arguments.push(MetalMixedHostArgument::Resident(
        PcuDeviceArgument::read_write(output[1], &mut r),
    ));
    assert!(matches!(
        prepared.call_mixed(&mut arguments),
        Err(PcuHostDispatchError::Backend(MetalError::ForeignSession))
    ));
    drop(arguments);
    assert!(!prepared.last_call_may_have_written());
    same(&read(&mut provider, &q), &[T::raw(9); 68]);
    same(&read(&mut provider, &r), &[T::raw(9); 70]);
}
#[test]
#[ignore = "required real Metal14 actual read/resident/host minima and private pre-read joint publication"]
fn fourteen_width_role_mixed_and_resident_joint_publication() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    macro_rules! types {($($ty:ty),+) => {$(qualify::<$ty>(&session, &foreign);)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

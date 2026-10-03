//! Actual unique owned/device resource projection and joint terminal publication.
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalDivRemRolePlan,
    MetalOwnedDispatchBackend,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDeviceArgument,
    PcuDeviceKernelBackend,
    PcuDispatchSubmission,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuMemoryPoolId,
    PcuOwnedCompletion,
    PcuOwnedDispatchBackend,
    PcuOwnedDispatchMemorySession,
    PcuPreparedDeviceKernel,
    PcuPreparedOwnedDispatch,
    PcuReproducibility,
    PcuValueType,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
};
#[rustfmt::skip]
use super::super::{
    mixed::{
        owner,
        owned_backend,
        read,
    },
    same,
    Sample,
};
fn qualify<T: Sample>(backend: &MetalOwnedDispatchBackend, foreign: &MetalOwnedDispatchBackend) {
    for profile in 0..6 {
        for portable in [false, true] {
            case::<T>(backend, foreign, profile, portable);
        }
    }
}
#[allow(clippy::too_many_lines)] // One retained owned+device pair checks projection, terminal faults, retry and post-drop lifetime.
fn case<T: Sample>(
    backend: &MetalOwnedDispatchBackend,
    foreign: &MetalOwnedDispatchBackend,
    profile: usize,
    portable: bool,
) {
    let (prepared, mut device, plan, unread) =
        super::graph::roles::fixture::<T, _>(65, profile, |ir| {
            let mut ir = *ir;
            if portable {
                ir.numerical_requirements.numerical_options.reproducibility =
                    PcuReproducibility::PortableV1;
            }
            let shape = PcuInvocationShape::invocations(
                core::num::NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
            );
            let plan = MetalDivRemRolePlan::assess(&ir).unwrap();
            let unread = ir
                .bindings
                .iter()
                .map(|binding| binding.reference())
                .find(|target| {
                    !plan.input_bindings().contains(target)
                        && !plan.output_bindings().contains(target)
                });
            (
                backend
                    .prepare_dispatch_owned_direct(
                        PcuDispatchSubmission { kernel: &ir, shape },
                        PcuInvocationParameters::empty(),
                    )
                    .unwrap(),
                backend.prepare_device_kernel(&ir).unwrap(),
                plan,
                unread,
            )
        });
    let requirements = prepared.binding_schema();
    assert_eq!(requirements.len(), plan.input_bindings().len() + 2);
    if let Some(unread) = unread {
        assert!(!requirements.iter().any(|binding| binding.target == unread));
    }
    let mut provider = backend.session().memory_provider(PcuMemoryPoolId(121));
    let mut other = foreign.session().memory_provider(PcuMemoryPoolId(121));
    let one = T::raw(4);
    let zero = T::raw(0);
    let sentinel = T::raw(9);
    let left = owner(&mut provider, &[one; 68]);
    let right = owner(&mut provider, &[one; 69]);
    let ignored = owner(&mut other, &[zero]);
    let mut q = owner(&mut provider, &[sentinel; 68]);
    let mut r = owner(&mut provider, &[sentinel; 70]);
    let outputs = plan.output_bindings();
    let mut arguments = vec![PcuDeviceArgument::read(plan.input_bindings()[0], &left)];
    if plan.input_bindings().len() == 2 {
        arguments.push(PcuDeviceArgument::read(plan.input_bindings()[1], &right));
    }
    if let Some(target) = unread {
        arguments.push(PcuDeviceArgument::read(target, &ignored));
    }
    arguments.push(PcuDeviceArgument::read_write(outputs[0], &mut q));
    arguments.push(PcuDeviceArgument::read_write(outputs[1], &mut r));
    device.call(&mut arguments).unwrap();
    drop(arguments);
    let expected_q = [vec![one; 65], vec![sentinel; 3]].concat();
    let expected_r = [vec![zero; 65], vec![sentinel; 5]].concat();
    same(&read(&mut provider, &q), &expected_q);
    same(&read(&mut provider, &r), &expected_r);
    let bind = |a: &pcu_facade::PcuDeviceBuffer<T, _>,
                b: &pcu_facade::PcuDeviceBuffer<T, _>,
                q: &pcu_facade::PcuDeviceBuffer<T, _>,
                r: &pcu_facade::PcuDeviceBuffer<T, _>| {
        let mut values = Vec::new();
        for (slot, &target) in plan.input_bindings().iter().enumerate() {
            values.push(
                backend
                    .bind(
                        target,
                        pcu_facade::PcuBindingAccess::ReadOnly,
                        PcuBindingType::Value(PcuValueType::Scalar(T::TYPE)),
                        [a, b][slot].resource(),
                    )
                    .unwrap(),
            );
        }
        for (slot, &target) in outputs.iter().enumerate() {
            values.push(
                backend
                    .bind(
                        target,
                        pcu_facade::PcuBindingAccess::ReadWrite,
                        PcuBindingType::Value(PcuValueType::Scalar(T::TYPE)),
                        [q, r][slot].resource(),
                    )
                    .unwrap(),
            );
        }
        values
    };
    let mut a = [one; 68];
    let mut b = [one; 69];
    if profile == 1 || profile == 5 {
        a[0] = zero;
    } else {
        a[2] = zero;
    }
    if profile == 4 {
        b[0] = zero;
    } else {
        b[2] = zero;
    }
    let bad_a = owner(&mut provider, &a);
    let bad_b = owner(&mut provider, &b);
    let mut failed = prepared.submit_owned(bind(&bad_a, &bad_b, &q, &r)).unwrap();
    let PcuCompletionOutcome::Fault(fault) = failed.wait().unwrap() else {
        panic!("missing joint zero divisor");
    };
    assert_eq!(fault.kind, pcu_facade::PcuExecutionFaultKind::DivideByZero);
    assert_eq!(
        fault.invocation_id,
        if matches!(profile, 1 | 4 | 5) { 0 } else { 2 }
    );
    assert!(!fault.recovered);
    same(&read(&mut provider, &q), &expected_q);
    same(&read(&mut provider, &r), &expected_r);
    let mut complete = prepared.submit_owned(bind(&left, &right, &q, &r)).unwrap();
    assert_eq!(complete.wait().unwrap(), PcuCompletionOutcome::Succeeded);
    drop(complete);
    drop(failed);
    drop(prepared);
    drop(device);
    same(&read(&mut provider, &q), &expected_q);
    same(&read(&mut provider, &r), &expected_r);
    same(&read(&mut other, &ignored), &[zero]);
}
#[test]
#[ignore = "required actual Metal14 owned/device role schema and transactional resident publication"]
fn fourteen_width_actual_role_owned_and_device_publication() {
    let backend = owned_backend();
    let foreign = owned_backend();
    macro_rules! types {($($ty:ty),+) => {$(qualify::<$ty>(&backend,&foreign);)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

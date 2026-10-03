//! Actual unique input roles and requested Portable headers stay separate from discovery claims.
#[path = "graph/graph.rs"]
mod graph;
#[path = "mixed/mixed.rs"]
mod mixed;
#[path = "owned/owned.rs"]
mod owned;
#[path = "portable_source/portable_source.rs"]
mod portable_source;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use super::{
    same,
    Sample,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalDivRemRolePlan,
    MetalError,
    MetalHostKernelError,
    MetalSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuReproducibility,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
};
struct Detached;
impl PcuHostKernelBackend for Detached {
    type Prepared = Self;
    type Error = MetalHostKernelError;
    fn prepare_host_kernel(&self, ir: &PcuDispatchKernelIr<'_>) -> Result<Self, Self::Error> {
        let plan = MetalDivRemRolePlan::assess(ir).map_err(PcuHostDispatchError::Backend)?;
        assert_eq!(plan.requirements(), ir.numerical_requirements);
        Ok(Self)
    }
}
impl PcuPreparedHostKernel for Detached {
    type Error = MetalHostKernelError;
    fn call(&mut self, _: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        Err(PcuHostDispatchError::Backend(MetalError::Unsupported))
    }
}
fn cold<T: Sample>() {
    for profile in 0..6 {
        graph::roles::fixture::<T, _>(65, profile, |ir| {
            let plan = MetalDivRemRolePlan::assess(ir).unwrap();
            assert_eq!(
                plan.input_bindings().len(),
                if profile == 2 || profile == 4 { 2 } else { 1 }
            );
            assert_eq!(plan.element_count(), 65);
            let clamp = PcuDispatchKernelIr {
                numerical_requirements: PcuImplementationRequirements {
                    range_policy: PcuRangePolicy::Clamp,
                    ..ir.numerical_requirements
                },
                ..*ir
            };
            assert!(MetalDivRemRolePlan::assess(&clamp).is_err());
            let mut portable = *ir;
            portable
                .numerical_requirements
                .numerical_options
                .reproducibility = PcuReproducibility::PortableV1;
            let retained = MetalDivRemRolePlan::assess(&portable).unwrap();
            assert_eq!(retained.requirements(), portable.numerical_requirements);
            assert_eq!(retained.input_bindings(), plan.input_bindings());
            assert_eq!(retained.input_element_counts(), plan.input_element_counts());
            assert_eq!(
                retained.implementation_local_id(),
                plan.implementation_local_id() + 0x1000
            );
        });
    }
    assert!(source::repeated_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::unread_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::reordered_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::grid_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::scalar_divisor_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(source::scalar_grid_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(portable_source::repeated_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(portable_source::unread_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(portable_source::reordered_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(portable_source::grid_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(portable_source::scalar_divisor_prepare::<T, 65, _>(&Detached).is_ok());
    assert!(portable_source::scalar_grid_prepare::<T, 65, _>(&Detached).is_ok());
}
#[test]
fn fourteen_width_detached_actual_read_roles_and_portable_source_headers() {
    macro_rules! types {($($ty:ty),+) => {$(cold::<$ty>();)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}
fn pairs<T: Sample>(
    profile: usize,
    mut call: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), MetalHostKernelError>,
) {
    let sentinel = T::raw(9);
    let mut q = [sentinel; 68];
    let mut r = q;
    for phase in 0..16 {
        let a: [T; 65] =
            core::array::from_fn(|lane| T::raw((phase + u8::try_from(lane % 16).unwrap()) % 16));
        let b: [T; 65] = core::array::from_fn(|lane| {
            T::raw((phase + u8::try_from(lane % 16).unwrap() + 4) % 16)
        });
        let mut expected_q = [sentinel; 68];
        let mut expected_r = expected_q;
        let mut fault = None;
        for lane in 0..65 {
            let (left, right) = match profile {
                0 => (a[lane], a[lane]),
                1 => (a[lane], a[0]),
                2 => (a[lane], b[lane]),
                3 => (a[0], a[lane]),
                4 => (a[lane], b[0]),
                5 => (a[0], a[0]),
                _ => unreachable!(),
            };
            match left
                .pcu_checked_div(right)
                .and_then(|q| left.pcu_checked_rem(right).map(|r| (q, r)))
            {
                Ok((q, r)) => {
                    expected_q[lane] = q;
                    expected_r[lane] = r;
                }
                Err(kind) => {
                    if fault.is_none() {
                        fault = Some((lane, kind));
                    }
                }
            }
        }
        let before = (q, r);
        let result = call(&a, &b, &mut q, &mut r);
        if let Some((lane, kind)) = fault {
            let Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(actual))) = result else {
                panic!("precise joint fault missing")
            };
            assert_eq!(actual.invocation_id, u64::try_from(lane).unwrap());
            assert_eq!(actual.kind, kind);
            assert!(!actual.recovered);
            same(&q, &before.0);
            same(&r, &before.1);
        } else {
            result.unwrap();
            same(&q, &expected_q);
            same(&r, &expected_r);
        }
        let before = (q, r);
        assert!(call(&a, &b, &mut q, &mut [sentinel; 64]).is_err());
        same(&q, &before.0);
        same(&r, &before.1);
        let ones = [T::raw(4); 65];
        call(&ones, &ones, &mut q, &mut r).unwrap();
        same(&q[..65], &ones);
        same(&r[..65], &[T::raw(0); 65]);
        same(&q[65..], &[sentinel; 3]);
        same(&r[65..], &[sentinel; 3]);
    }
}
fn native<T: Sample>(session: &MetalSession) {
    let backend = session.checked_div_rem_role_backend();
    macro_rules! run {
        ($source:ident) => {{
            let mut p = $source::repeated_prepare::<T, 65, _>(&backend).unwrap();
            pairs::<T>(0, |a, _, q, r| p(q, a, r));
            let mut p = $source::unread_prepare::<T, 65, _>(&backend).unwrap();
            pairs::<T>(1, |a, _, q, r| p(&[], r, a, q));
            let mut p = $source::reordered_prepare::<T, 65, _>(&backend).unwrap();
            pairs::<T>(2, |a, b, q, r| p(b, r, a, q));
            let mut p = $source::grid_prepare::<T, 65, _>(&backend).unwrap();
            pairs::<T>(3, |a, _, q, r| p(&[], r, a, q));
            let mut p = $source::scalar_divisor_prepare::<T, 65, _>(&backend).unwrap();
            pairs::<T>(4, |a, b, q, r| p(q, &b[0], r, a));
            let mut p = $source::scalar_grid_prepare::<T, 65, _>(&backend).unwrap();
            pairs::<T>(5, |a, _, q, r| p(&a[0], q, r));
        }};
    }
    run!(source);
    run!(portable_source);
}
#[test]
#[ignore = "required real Metal fourteen-width actual read/broadcast source and requested Portable joint faults"]
fn fourteen_width_actual_read_source_and_portable_joint_publication() {
    let session = MetalSession::open(0).unwrap();
    macro_rules! types {($($ty:ty),+) => {$(native::<$ty>(&session);)+};}
    types!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

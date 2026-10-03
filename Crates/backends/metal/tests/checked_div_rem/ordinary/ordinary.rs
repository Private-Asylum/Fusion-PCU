//! Genuine four-argument ordinary resident source and immutable failure publication.
#[rustfmt::skip]
use pcu_facade::{global,PcuTensor,PcuMemoryPoolId,PcuScalar,PcuExecutionError,PcuExecutionFaultKind};
use super::{Sample, same, source, mixed};
fn owner<T: PcuScalar>(
    backend: &fusion_pcu_metal::MetalOwnedDispatchBackend,
    values: &[T],
) -> PcuTensor<T> {
    PcuTensor::from_device_buffer(
        backend.clone(),
        backend.upload_buffer(PcuMemoryPoolId(122), values).unwrap(),
        &[values.len()],
    )
    .unwrap()
}
#[test]
#[ignore = "Requires actual Metal ordinary fourteen-width four-argument host/resident/mixed joint publication."]
fn fourteen_width_ordinary_resident_joint_publication() {
    let backend = mixed::owned_backend();
    for mode in [
        pcu_facade::PcuNumericalMode::Boundary,
        pcu_facade::PcuNumericalMode::Strict,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Metal,
            numerical_mode: mode,
            ..Default::default()
        })
        .unwrap();
        qualify::<i8>(&backend);
        qualify::<u8>(&backend);
        qualify::<i16>(&backend);
        qualify::<u16>(&backend);
        qualify::<i32>(&backend);
        qualify::<u32>(&backend);
        qualify::<i64>(&backend);
        qualify::<u64>(&backend);
        qualify::<i128>(&backend);
        qualify::<u128>(&backend);
        qualify::<super::PcuI256>(&backend);
        qualify::<super::PcuU256>(&backend);
        qualify::<super::PcuI512>(&backend);
        qualify::<super::PcuU512>(&backend);
        global::clear_thread_cache().unwrap();
    }
    global::use_defaults().unwrap();
}

fn qualify<T: Sample>(backend: &fusion_pcu_metal::MetalOwnedDispatchBackend) {
    let zero = T::raw(0);
    let one = T::raw(4);
    let sentinel = T::raw(9);
    let lhs = owner(backend, &[one; 5]);
    let rhs = owner(backend, &[one; 5]);
    let mut q = owner(backend, &[sentinel; 5]);
    let mut r = owner(backend, &[sentinel; 5]);
    source::generic::direct::<T, 3>(&lhs, &rhs, &mut q, &mut r).unwrap();
    let mut actual = [zero; 5];
    q.read_into(&mut actual).unwrap();
    same(&actual, &[one, one, one, sentinel, sentinel]);
    r.read_into(&mut actual).unwrap();
    same(&actual, &[zero, zero, zero, sentinel, sentinel]);
    let bad = owner(backend, &[one, one, zero, one, one]);
    assert!(
        matches!(source::generic::direct::<T,3>(&lhs,&bad,&mut q,&mut r),Err(PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id==2 && fault.kind==PcuExecutionFaultKind::DivideByZero)
    );
    q.read_into(&mut actual).unwrap();
    same(&actual, &[one, one, one, sentinel, sentinel]);
    r.read_into(&mut actual).unwrap();
    same(&actual, &[zero, zero, zero, sentinel, sentinel]);
    let mut host = [sentinel; 5];
    source::generic::direct::<T, 3>(&lhs, &[one; 3], &mut host, &mut r).unwrap();
    same(&host, &[one, one, one, sentinel, sentinel]);
    host.fill(sentinel);
    source::generic::grid::<T, 3>(&[one; 3], &rhs, &mut q, &mut host).unwrap();
    same(&host, &[zero, zero, zero, sentinel, sentinel]);
    assert!(source::generic::direct::<T, 3>(&lhs, &rhs, &mut q, &mut []).is_err());
    q.read_into(&mut actual).unwrap();
    same(&actual, &[one, one, one, sentinel, sentinel]);
    let foreign = owner(&mixed::owned_backend(), &[sentinel; 5]);
    let mut foreign = foreign;
    assert!(matches!(
        source::generic::direct::<T, 3>(&lhs, &rhs, &mut q, &mut foreign),
        Err(PcuExecutionError::Argument(
            pcu_facade::PcuArgumentError::SessionMismatch
        ))
    ));
    foreign.read_into(&mut actual).unwrap();
    same(&actual, &[sentinel; 5]);
    q.read_into(&mut actual).unwrap();
    same(&actual, &[one, one, one, sentinel, sentinel]);
    source::generic::direct::<T, 3>(&lhs, &rhs, &mut q, &mut r).unwrap();
    drop(r);
    drop(rhs);
    drop(lhs);
    q.read_into(&mut actual).unwrap();
    same(&actual, &[one, one, one, sentinel, sentinel]);
}

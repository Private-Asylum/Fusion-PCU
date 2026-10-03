//! Genuine fourteen-width ordinary source across host/resident/mixed immutable affinities.
#[rustfmt::skip]
use pcu_facade::{PcuTensor,PcuMemoryPoolId,PcuRuntimeDiscovery,PcuExecutionError,PcuArgumentError,
    PcuU256,PcuI256,PcuU512,PcuI512};
use super::{source, Sample, same};
fn backend() -> fusion_pcu_metal::MetalOwnedDispatchBackend {
    let discovery = fusion_pcu_metal::MetalDiscovery::discover().unwrap();
    let mut providers = [pcu_facade::PcuProviderDescriptor {
        id: pcu_facade::PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: pcu_facade::PcuProviderReadiness {
            status: pcu_facade::PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    discovery.providers(&mut providers).unwrap();
    discovery
        .open_owned_device(pcu_facade::PcuObjectRef {
            provider: providers[0].id,
            generation: providers[0].generation,
            kind: pcu_facade::PcuObjectKind::Device,
            id: 0,
        })
        .unwrap()
}
fn owner<T: Sample>(
    backend: &fusion_pcu_metal::MetalOwnedDispatchBackend,
    input: &[T],
) -> PcuTensor<T> {
    let data = backend.upload_buffer(PcuMemoryPoolId(114), input).unwrap();
    PcuTensor::from_device_buffer(backend.clone(), data, &[input.len()]).unwrap()
}
fn qualify<T: Sample>() {
    let backend = backend();
    let zero = T::raw(0);
    let one = T::raw(4);
    let sentinel = T::raw(5);
    let left = owner(&backend, &[zero, one, one, sentinel, sentinel]);
    let right = owner(&backend, &[one, one, zero, sentinel, sentinel]);
    let mut output = owner(&backend, &[sentinel; 5]);
    source::add::<T, 3>(&left, &right, &mut output).unwrap();
    let expected = [
        one,
        one.pcu_checked_add(one).unwrap(),
        one,
        sentinel,
        sentinel,
    ];
    let mut actual = [zero; 5];
    output.read_into(&mut actual).unwrap();
    same(&actual, &expected);
    let mut host = [sentinel; 5];
    source::add::<T, 3>(&left, &[one, one, zero], &mut host).unwrap();
    same(&host, &expected);
    source::mul::<T, 3>(&[one, one, zero], &right, &mut output).unwrap();
    output.read_into(&mut actual).unwrap();
    same(&actual, &[one, one, zero, sentinel, sentinel]);
    let limit = if T::raw(1).pcu_checked_add(one).is_err() {
        T::raw(1)
    } else {
        T::raw(3)
    };
    let bad = owner(&backend, &[limit; 5]);
    let mut discarded = owner(&backend, &[sentinel; 5]);
    assert!(source::add::<T, 3>(&bad, &[one; 3], &mut discarded).is_err());
    assert!(matches!(
        discarded.read_into(&mut actual),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    let mut recovered = owner(&backend, &[sentinel; 5]);
    let result = source::add_clamp::<T, 3>(&bad, &[one; 3], &mut recovered);
    assert!(matches!(result,Err(PcuExecutionError::ArithmeticFault(fault))if fault.recovered));
    recovered.read_into(&mut actual).unwrap();
    same(&actual, &[limit, limit, limit, sentinel, sentinel]);
    source::sub::<T, 3>(&output, &[zero; 3], &mut host).unwrap();
    same(&host, &[one, one, zero, sentinel, sentinel]);
    let foreign = owner(&self::backend(), &[one; 5]);
    let before = host;
    assert!(matches!(
        source::add::<T, 3>(&left, &foreign, &mut host),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::SessionMismatch
        ))
    ));
    same(&host, &before);
    drop(output);
    drop(right);
    drop(backend);
    left.read_into(&mut actual).unwrap();
    same(&actual, &[zero, one, one, sentinel, sentinel]);
}
#[test]
#[ignore = "Requires actual ordinary Metal14 host/resident/mixed source, complete recovery/discard and retained affinity."]
fn fourteen_width_ordinary_resident_and_mixed_publication() {
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    macro_rules! all{($($ty:ty),+)=>{$(qualify::<$ty>();)+};}
    all!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
    pcu_facade::global::clear_thread_cache().unwrap();
}

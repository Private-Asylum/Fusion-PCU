//! Actual 64-bit annotated arithmetic versus independent sealed scalar reference results.
use pcu_facade::pcu;
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedInteger,
    PcuExecutionFaultKind,
    PcuHostDispatchError,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalError,
    MetalSession,
};

macro_rules! source_maps {
    ($add:ident, $sub:ident, $mul:ident, $identity:ident, $ty:ty) => {
        #[pcu(invocations = N, crate_path = ::pcu_facade)]
        fn $add<const N: usize>(lhs: &[$ty], rhs: &[$ty], output: &mut [$ty]) {
            let id = pcu::context::global_invocation_id();
            output[id] = lhs[id] + rhs[id];
        }
        #[pcu(invocations = N, flag(strict), crate_path = ::pcu_facade)]
        fn $sub<const N: usize>(lhs: &[$ty], rhs: &[$ty], output: &mut [$ty]) {
            let id = pcu::context::global_invocation_id();
            output[id] = lhs[id] - rhs[id];
        }
        #[pcu(invocations = N, crate_path = ::pcu_facade)]
        fn $mul<const N: usize>(lhs: &[$ty], rhs: &[$ty], output: &mut [$ty]) {
            let id = pcu::context::global_invocation_id();
            output[id] = lhs[id] * rhs[id];
        }
        #[pcu(invocations = N, crate_path = ::pcu_facade)]
        fn $identity<const N: usize>(input: &[$ty], output: &mut [$ty]) {
            let id = pcu::context::global_invocation_id();
            output[id] = input[id];
        }
    };
}
source_maps!(signed_add, signed_sub, signed_mul, signed_identity, i64);
source_maps!(
    unsigned_add,
    unsigned_sub,
    unsigned_mul,
    unsigned_identity,
    u64
);

fn corpus<T: PcuCheckedInteger + std::fmt::Debug + Eq>(
    edges: &[T],
    sentinel: T,
    source: &mut impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuHostDispatchError<MetalError>>,
    oracle: impl Fn(T, T) -> Result<T, PcuExecutionFaultKind>,
) {
    let mut output = [sentinel; 3];
    for &left in edges {
        for &right in edges {
            let before = output;
            match oracle(left, right) {
                Ok(expected) => {
                    source(&[left], &[right], &mut output).unwrap();
                    assert_eq!(
                        output,
                        [expected, sentinel, sentinel],
                        "left={left:?}, right={right:?}"
                    );
                }
                Err(kind) => {
                    let Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) =
                        source(&[left], &[right], &mut output)
                    else {
                        panic!("missing 64-bit range fault: {left:?}, {right:?}");
                    };
                    assert_eq!(fault.kind, kind, "left={left:?}, right={right:?}");
                    assert_eq!(fault.invocation_id, 0);
                    assert!(!fault.recovered);
                    assert_eq!(output, before);
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual Metal I64/U64 source GPU qualification."]
fn complete_edge_pairs_and_retained_identity_owners() {
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut sa = signed_add_prepare::<1, _>(&session).unwrap();
    let mut ss = signed_sub_prepare::<1, _>(&session).unwrap();
    let mut sm = signed_mul_prepare::<1, _>(&session).unwrap();
    let mut ua = unsigned_add_prepare::<1, _>(&session).unwrap();
    let mut us = unsigned_sub_prepare::<1, _>(&session).unwrap();
    let mut um = unsigned_mul_prepare::<1, _>(&session).unwrap();
    let mut si = signed_identity_prepare::<3, _>(&session).unwrap();
    let mut ui = unsigned_identity_prepare::<3, _>(&session).unwrap();
    drop(session);
    let signed = [
        i64::MIN,
        i64::MIN + 1,
        -4_294_967_296,
        -3_037_000_500,
        -3_037_000_499,
        -2,
        -1,
        0,
        1,
        2,
        3_037_000_499,
        3_037_000_500,
        4_294_967_296,
        i64::MAX - 1,
        i64::MAX,
    ];
    let unsigned = [
        0,
        1,
        2,
        3,
        65_535,
        65_536,
        4_294_967_295,
        4_294_967_296,
        4_294_967_297,
        1 << 63,
        u64::MAX - 1,
        u64::MAX,
    ];
    corpus(&signed, 91, &mut sa, i64::pcu_checked_add);
    corpus(&signed, 91, &mut ss, i64::pcu_checked_sub);
    corpus(&signed, 91, &mut sm, i64::pcu_checked_mul);
    corpus(&unsigned, 91, &mut ua, u64::pcu_checked_add);
    corpus(&unsigned, 91, &mut us, u64::pcu_checked_sub);
    corpus(&unsigned, 91, &mut um, u64::pcu_checked_mul);
    let mut signed_output = [91_i64; 5];
    si(&[i64::MIN, -1, i64::MAX], &mut signed_output).unwrap();
    assert_eq!(signed_output, [i64::MIN, -1, i64::MAX, 91, 91]);
    let mut unsigned_output = [91_u64; 5];
    ui(&[u64::MAX, 1 << 63, 0], &mut unsigned_output).unwrap();
    assert_eq!(unsigned_output, [u64::MAX, 1 << 63, 0, 91, 91]);
}
#[test]
#[ignore = "Requires actual ordinary Metal I64/U64 source routing."]
fn ordinary_global_checked_64_bit_fault_order_and_retry() {
    let _policy_guard = crate::source_policy_guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut signed = [91_i64; 5];
    signed_mul::<3>(&[i64::MIN, -3, 4_294_967_296], &[1, -2, 2], &mut signed).unwrap();
    assert_eq!(signed, [i64::MIN, 6, 8_589_934_592, 91, 91]);
    let before = signed;
    let Err(pcu_facade::PcuExecutionError::ArithmeticFault(fault)) =
        signed_mul::<3>(&[1, i64::MIN, i64::MAX], &[2, -1, 2], &mut signed)
    else {
        panic!("missing signed 64-bit global fault");
    };
    assert_eq!(fault.invocation_id, 1);
    assert_eq!(signed, before);
    let mut unsigned = [91_u64; 5];
    unsigned_add::<3>(&[0, 4_294_967_295, u64::MAX - 1], &[1, 1, 1], &mut unsigned).unwrap();
    assert_eq!(unsigned, [1, 4_294_967_296, u64::MAX, 91, 91]);
    let before = unsigned;
    let Err(pcu_facade::PcuExecutionError::ArithmeticFault(fault)) =
        unsigned_add::<3>(&[1, u64::MAX, u64::MAX], &[1, 1, 1], &mut unsigned)
    else {
        panic!("missing unsigned 64-bit global fault");
    };
    assert_eq!(fault.invocation_id, 1);
    assert_eq!(unsigned, before);
    unsigned_add::<3>(&[1, 2, 3], &[4, 5, 6], &mut unsigned).unwrap();
    assert_eq!(unsigned, [5, 7, 9, 91, 91]);
}

#[test]
#[ignore = "Requires actual Metal typed U64 resident source qualification."]
fn resident_64_bit_fault_priority_tail_extent_and_retry() {
    use pcu_facade::PcuRuntimeDiscovery;
    let _policy_guard = crate::source_policy_guard();
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
    let backend = discovery
        .open_owned_device(pcu_facade::PcuObjectRef {
            provider: providers[0].id,
            generation: providers[0].generation,
            kind: pcu_facade::PcuObjectKind::Device,
            id: 0,
        })
        .unwrap();
    let pool = pcu_facade::PcuMemoryPoolId(83);
    let left = backend
        .upload_buffer(pool, &[1_u64, 0, u64::MAX, 81, 82])
        .unwrap();
    let right = backend
        .upload_buffer(pool, &[1_u64, 1, u64::MAX - 1, 83, 84])
        .unwrap();
    let mut output = backend.upload_buffer(pool, &[91_u64; 5]).unwrap();
    let mut source = unsigned_sub_prepare_device::<3, _>(&backend).unwrap();
    let Err(fusion_pcu_metal::MetalOwnedDispatchError::Metal(MetalError::Arithmetic(fault))) =
        source(&left, &right, &mut output)
    else {
        panic!("missing resident U64 underflow");
    };
    assert_eq!(fault.invocation_id, 1);
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
    let mut host = [0_u64; 5];
    backend.download_buffer(pool, &output, &mut host).unwrap();
    assert_eq!(&host[3..], &[91, 91]);
    let right = backend
        .upload_buffer(pool, &[1_u64, 0, u64::MAX - 1, 83, 84])
        .unwrap();
    source(&left, &right, &mut output).unwrap();
    backend.download_buffer(pool, &output, &mut host).unwrap();
    assert_eq!(host, [0, 0, 1, 91, 91]);
    let short = pcu_facade::PcuDeviceBuffer::<u64, _>::new(left.into_resource(), 2);
    assert!(matches!(
        source(&short, &right, &mut output),
        Err(fusion_pcu_metal::MetalOwnedDispatchError::Binding(
            pcu_facade::PcuOwnedDispatchBindingError::BufferTooSmall { .. }
        ))
    ));
    drop(discovery);
}

#[test]
#[ignore = "Requires actual Metal I64/U64 logical owner construction, readback and same-session source proof."]
fn signed_unsigned_logical_owner_import_readback_and_mixed_source() {
    let _policy_guard = crate::source_policy_guard();
    let pool = pcu_facade::PcuMemoryPoolId(93);
    let backend = super::low_precision::open_backend();
    let left = backend.upload_buffer(pool, &[1_i64, 2, 3, 81, 82]).unwrap();
    let right = backend.upload_buffer(pool, &[5_i64, 6, 7, 83, 84]).unwrap();
    let output = backend.upload_buffer(pool, &[91_i64; 5]).unwrap();
    let left = pcu_facade::PcuTensor::from_device_buffer(backend.clone(), left, &[5]).unwrap();
    let right = pcu_facade::PcuTensor::from_device_buffer(backend.clone(), right, &[5]).unwrap();
    let mut output = pcu_facade::PcuTensor::from_device_buffer(backend, output, &[5]).unwrap();
    let mut host = [0_i64; 5];
    left.read_into(&mut host).unwrap();
    assert_eq!(host, [1, 2, 3, 81, 82]);
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    signed_add::<3>(&left, &right, &mut output).unwrap();
    output.read_into(&mut host).unwrap();
    assert_eq!(host, [6, 8, 10, 91, 91]);
    signed_add::<3>(&left, &[5, 6, 7], &mut host).unwrap();
    assert_eq!(host, [6, 8, 10, 91, 91]);
    let backend = super::low_precision::open_backend();
    let left = backend.upload_buffer(pool, &[1_u64, 2, 3, 81, 82]).unwrap();
    let output = backend.upload_buffer(pool, &[91_u64; 5]).unwrap();
    let left = pcu_facade::PcuTensor::from_device_buffer(backend.clone(), left, &[5]).unwrap();
    let mut output = pcu_facade::PcuTensor::from_device_buffer(backend, output, &[5]).unwrap();
    let mut host = [0_u64; 5];
    left.read_into(&mut host).unwrap();
    assert_eq!(host, [1, 2, 3, 81, 82]);
    unsigned_add::<3>(&left, &[5, 6, 7], &mut output).unwrap();
    output.read_into(&mut host).unwrap();
    assert_eq!(host, [6, 8, 10, 91, 91]);
    unsigned_sub::<3>(&output, &[5, 6, 7], &mut host).unwrap();
    assert_eq!(host, [1, 2, 3, 91, 91]);
    pcu_facade::global::use_defaults().unwrap();
}

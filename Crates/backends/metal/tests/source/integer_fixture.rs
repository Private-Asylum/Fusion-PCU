//! Actual annotated source against the explicit Metal host preparation seam.

use pcu_facade::pcu;
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalError,
    MetalSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuExecutionFaultKind,
    PcuHostDispatchError,
};

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn checked_add<const N: usize>(lhs: &[u32], rhs: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] + rhs[id];
}

#[test]
#[ignore = "Requires actual macOS Metal device and runs annotated source on GPU."]
fn annotated_source_success_fault_retry_and_tail_preservation() {
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut source = checked_add_prepare::<3, _>(&session).unwrap();
    let mut output = [91_u32; 5];
    source(&[1, 2, 3], &[3, 4, 5], &mut output).unwrap();
    assert_eq!(output, [4, 6, 8, 91, 91]);
    let before = output;
    assert!(
        matches!(source(&[1, u32::MAX, u32::MAX], &[3, 1, 1], &mut output), Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.invocation_id == 1 && !fault.recovered)
    );
    assert_eq!(output, before);
    source(&[7, 8, 9], &[1, 2, 3], &mut output).unwrap();
    assert_eq!(output, [8, 10, 12, 91, 91]);
    assert!(matches!(
        source(&[1], &[1, 2, 3], &mut output),
        Err(PcuHostDispatchError::BufferTooSmall(_))
    ));
    assert_eq!(output, [8, 10, 12, 91, 91]);
    drop(session);
    source(&[2, 2, 2], &[3, 3, 3], &mut output).unwrap();
    assert_eq!(output, [5, 5, 5, 91, 91]);
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn signed_add<const N: usize>(lhs: &[i32], rhs: &[i32], output: &mut [i32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] + rhs[id];
}
#[pcu(invocations = N, flag(strict), crate_path = ::pcu_facade)]
fn signed_sub<const N: usize>(lhs: &[i32], rhs: &[i32], output: &mut [i32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] - rhs[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn signed_mul<const N: usize>(lhs: &[i32], rhs: &[i32], output: &mut [i32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] * rhs[id];
}

fn signed_corpus(
    source: &mut impl FnMut(&[i32], &[i32], &mut [i32]) -> Result<(), PcuHostDispatchError<MetalError>>,
    oracle: impl Fn(i32, i32) -> Result<i32, PcuExecutionFaultKind>,
) {
    let edges = [
        i32::MIN,
        i32::MIN + 1,
        -65_536,
        -46_341,
        -46_340,
        -2,
        -1,
        0,
        1,
        2,
        46_340,
        46_341,
        65_536,
        i32::MAX - 1,
        i32::MAX,
    ];
    let mut output = [91_i32; 3];
    for left in edges {
        for right in edges {
            let before = output;
            match oracle(left, right) {
                Ok(expected) => {
                    source(&[left], &[right], &mut output).unwrap();
                    assert_eq!(output, [expected, 91, 91], "left={left}, right={right}");
                }
                Err(kind) => {
                    let Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) =
                        source(&[left], &[right], &mut output)
                    else {
                        panic!("missing signed fault: {left}, {right}");
                    };
                    assert_eq!(fault.kind, kind, "left={left}, right={right}");
                    assert_eq!(fault.invocation_id, 0);
                    assert!(!fault.recovered);
                    assert_eq!(output, before);
                }
            }
        }
    }
}

#[test]
#[ignore = "Requires actual Metal I32 checked source hardware qualification."]
fn signed_source_full_edge_pair_oracles_and_retained_owner() {
    use pcu_facade::PcuCheckedInteger;
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut add = signed_add_prepare::<1, _>(&session).unwrap();
    let mut sub = signed_sub_prepare::<1, _>(&session).unwrap();
    let mut mul = signed_mul_prepare::<1, _>(&session).unwrap();
    drop(session);
    signed_corpus(&mut add, i32::pcu_checked_add);
    signed_corpus(&mut sub, i32::pcu_checked_sub);
    signed_corpus(&mut mul, i32::pcu_checked_mul);
}

#[test]
#[ignore = "Requires actual Metal I32 resident source hardware qualification."]
fn signed_device_fault_priority_retry_and_tail() {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuOwnedDispatchMemorySession,
        PcuRuntimeDiscovery,
    };
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
    let pool = pcu_facade::PcuMemoryPoolId(77);
    let left = backend
        .upload_buffer(pool, &[i32::MIN, 2, i32::MAX, 78, 79])
        .unwrap();
    let right = backend
        .upload_buffer(pool, &[-1_i32, 3, 2, 81, 82])
        .unwrap();
    let mut output = backend.upload_buffer(pool, &[91_i32; 5]).unwrap();
    let mut source = signed_mul_prepare_device::<3, _>(&backend).unwrap();
    let Err(fusion_pcu_metal::MetalOwnedDispatchError::Metal(MetalError::Arithmetic(fault))) =
        source(&left, &right, &mut output)
    else {
        panic!("missing resident overflow");
    };
    assert_eq!(fault.invocation_id, 0);
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    let mut host = [0_i32; 5];
    backend.download_buffer(pool, &output, &mut host).unwrap();
    assert_eq!(&host[3..], &[91, 91]);
    let right = backend
        .upload_buffer(pool, &[1_i32, 3, -1, 81, 82])
        .unwrap();
    source(&left, &right, &mut output).unwrap();
    backend.download_buffer(pool, &output, &mut host).unwrap();
    assert_eq!(host, [i32::MIN, 6, -i32::MAX, 91, 91]);
    let provider = backend.memory_provider(pool);
    drop((discovery, backend));
    source(&left, &right, &mut output).unwrap();
    drop(provider);
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn signed_identity<const N: usize>(input: &[i32], output: &mut [i32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}
#[test]
#[ignore = "Requires actual Metal I32 identity source hardware qualification."]
fn signed_identity_exact_bits_and_tail() {
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut source = signed_identity_prepare::<3, _>(&session).unwrap();
    let mut output = [91; 5];
    source(&[i32::MIN, -1, i32::MAX], &mut output).unwrap();
    assert_eq!(output, [i32::MIN, -1, i32::MAX, 91, 91]);
    source(&[0, 1, -2], &mut output).unwrap();
    assert_eq!(output, [0, 1, -2, 91, 91]);
}

#[test]
#[ignore = "Requires actual Metal ordinary global I32 source routing."]
fn ordinary_global_signed_source_success_fault_and_retry() {
    let _policy_guard = crate::source_policy_guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut output = [91_i32; 5];
    signed_mul::<3>(&[i32::MIN, -7, 12], &[1, -3, 4], &mut output).unwrap();
    assert_eq!(output, [i32::MIN, 21, 48, 91, 91]);
    let before = output;
    let Err(pcu_facade::PcuExecutionError::ArithmeticFault(fault)) =
        signed_mul::<3>(&[1, i32::MIN, i32::MAX], &[2, -1, 2], &mut output)
    else {
        panic!("missing ordinary signed fault");
    };
    assert_eq!(fault.invocation_id, 1);
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(output, before);
    signed_mul::<3>(&[-2, -3, -4], &[5, 6, 7], &mut output).unwrap();
    assert_eq!(output, [-10, -18, -28, 91, 91]);
}

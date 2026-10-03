//! Actual integer-synthesized binary64 against the independent core U128 reference.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuDispatchFloatBinaryOp as Op,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy as Policy,
    PcuHostArgument,
    PcuBindingRef,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalBuffer,
    MetalError,
    MetalSession,
};

pub fn oracle(op: Op, policy: Policy, left: f64, right: f64) -> Result<f64, PcuExecutionFaultKind> {
    match op {
        Op::Add => left.pcu_checked_add_with_policy(right, policy),
        Op::Sub => left.pcu_checked_sub_with_policy(right, policy),
        Op::Mul => left.pcu_checked_mul_with_policy(right, policy),
        Op::Div => left.pcu_checked_div_with_policy(right, policy),
    }
}
fn upload(session: &MetalSession, input: &[f64]) -> MetalBuffer {
    session
        .upload_bytes(PcuHostArgument::read(PcuBindingRef::new(0, 0), input).bytes())
        .unwrap()
}
fn bits(buffer: &MetalBuffer) -> Vec<u64> {
    buffer
        .download_u32()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|words| u64::from(words[0]) | (u64::from(words[1]) << 32))
        .collect()
}
const fn random(seed: &mut u64) -> f64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let bits = if *seed & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 {
        *seed ^ 0x0010_0000_0000_0000
    } else {
        *seed
    };
    f64::from_bits(bits)
}

#[test]
#[ignore = "Requires actual M4 checked F64 integer GPU qualification."]
fn all_operations_policies_edges_and_98304_success_bits_match_core() {
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let edges = [
        0,
        0x8000_0000_0000_0000,
        1,
        2,
        3,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x0010_0000_0000_0001,
        0x3ca0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0001,
        0x3ff0_0000_0000_0002,
        0x4000_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff8_0000_0000_0000,
        0x8000_0000_0000_0001,
    ]
    .map(f64::from_bits);
    for policy in [
        Policy::IeeeAfterRounding,
        Policy::RejectSubnormalResult,
        Policy::AllowGradualUnderflow,
    ] {
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
            let map = session.prepare_f64_binary(op, policy).unwrap();
            for left in edges {
                for right in edges {
                    let actual =
                        map.execute(&upload(&session, &[left]), &upload(&session, &[right]));
                    match oracle(op, policy, left, right) {
                        Ok(expected) => assert_eq!(
                            bits(&actual.unwrap()),
                            [expected.to_bits()],
                            "{op:?}/{policy:?}: {left:?}, {right:?}"
                        ),
                        Err(kind) => {
                            let Err(MetalError::Arithmetic(fault)) = actual else {
                                panic!("missing {op:?}/{policy:?} fault: {left:?}, {right:?}");
                            };
                            assert_eq!(fault.kind, kind, "{op:?}/{policy:?}: {left:?}, {right:?}");
                            assert_eq!(fault.invocation_id, 0);
                            assert!(!fault.recovered);
                        }
                    }
                }
            }
            let mut seed = 0x9e37_79b9_7f4a_7c15;
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut expected = Vec::new();
            while left.len() < 8192 {
                let a = random(&mut seed);
                let b = random(&mut seed);
                if let Ok(value) = oracle(op, policy, a, b) {
                    left.push(a);
                    right.push(b);
                    expected.push(value.to_bits());
                }
            }
            let result = map
                .execute(&upload(&session, &left), &upload(&session, &right))
                .unwrap();
            assert_eq!(
                bits(&result),
                expected,
                "{op:?}/{policy:?} randomized finite corpus"
            );
            assert!(matches!(
                map.execute(
                    &session.upload_u32(&[0]).unwrap(),
                    &session.upload_u32(&[0]).unwrap()
                ),
                Err(MetalError::InvalidExtent)
            ));
            let foreign = MetalSession::open(0).unwrap();
            assert!(matches!(
                map.execute(&upload(&foreign, &[1.0]), &upload(&session, &[1.0])),
                Err(MetalError::ForeignSession)
            ));
        }
    }
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn add<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, flag(strict), crate_path = ::pcu_facade)]
pub fn sub<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn mul<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn div<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}

#[test]
#[ignore = "Requires actual checked F64 source preparation and ordinary Metal routing."]
fn genuine_source_default_strict_fault_priority_tail_and_retry() {
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut add = add_prepare::<3, _>(&session).unwrap();
    let mut sub = sub_prepare::<3, _>(&session).unwrap();
    let mut mul = mul_prepare::<3, _>(&session).unwrap();
    let mut div = div_prepare::<3, _>(&session).unwrap();
    let mut output = [91.0_f64; 5];
    add(
        &[1.0, -0.0, f64::from_bits(1)],
        &[2.0, -0.0, 0.0],
        &mut output,
    )
    .unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [3.0, -0.0, f64::from_bits(1), 91.0, 91.0].map(f64::to_bits)
    );
    sub(&[1.0, 0.0, 3.0], &[2.0, 0.0, 1.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [-1.0_f64, 0.0, 2.0, 91.0, 91.0].map(f64::to_bits)
    );
    mul(&[1.0, -2.0, 3.0], &[2.0, 3.0, 4.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [2.0_f64, -6.0, 12.0, 91.0, 91.0].map(f64::to_bits)
    );
    let before = output.map(f64::to_bits);
    let Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) =
        div(&[1.0, 1.0, f64::NAN], &[2.0, 0.0, 1.0], &mut output)
    else {
        panic!("missing earliest divide fault");
    };
    assert_eq!(fault.invocation_id, 1);
    assert_eq!(fault.kind, PcuExecutionFaultKind::DivideByZero);
    assert_eq!(output.map(f64::to_bits), before);
    div(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [0.5, 0.5, 0.5, 91.0, 91.0].map(f64::to_bits)
    );
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    self::add::<3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [5.0, 7.0, 9.0, 91.0, 91.0].map(f64::to_bits)
    );
    pcu_facade::global::use_defaults().unwrap();
}

#[pcu(invocations = N, flag(allow_gradual_underflow), crate_path = ::pcu_facade)]
fn gradual<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, flag(reject_subnormal_result), crate_path = ::pcu_facade)]
fn tight<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = 4, crate_path = ::pcu_facade)]
fn grid_div<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn swapped<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = right[id] - left[id];
}
#[test]
#[ignore = "Requires actual F64 source underflow, swapped SSA, grid and negative policy proof."]
fn source_underflow_profiles_swapped_grid_and_portable_refusal() {
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut gradual = gradual_prepare::<3, _>(&session).unwrap();
    let mut tight = tight_prepare::<3, _>(&session).unwrap();
    let mut normal = mul_prepare::<3, _>(&session).unwrap();
    let mut grid = grid_div_prepare::<3, _>(&session).unwrap();
    let mut swapped = swapped_prepare::<3, _>(&session).unwrap();
    let mut output = [91.0_f64; 5];
    let tiny = f64::from_bits(1);
    gradual(&[tiny, -tiny, 1.0], &[0.5, 0.5, 2.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [0.0_f64, -0.0, 2.0, 91.0, 91.0].map(f64::to_bits)
    );
    let before = output.map(f64::to_bits);
    assert!(
        matches!(normal(&[1.0, tiny, 1.0], &[1.0, 0.5, 1.0], &mut output), Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) if fault.invocation_id == 1 && fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(output.map(f64::to_bits), before);
    assert!(
        matches!(tight(&[tiny, 1.0, 1.0], &[1.0; 3], &mut output), Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) if fault.invocation_id == 0 && fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(output.map(f64::to_bits), before);
    grid(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [0.5_f64, 0.5, 0.5, 91.0, 91.0].map(f64::to_bits)
    );
    swapped(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [3.0_f64, 3.0, 3.0, 91.0, 91.0].map(f64::to_bits)
    );
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        numerical_options: pcu_facade::PcuNumericalOptions {
            reproducibility: pcu_facade::PcuReproducibility::PortableV1,
            ..pcu_facade::PcuNumericalOptions::default()
        },
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    assert!(matches!(
        add::<3>(&[1.0; 3], &[2.0; 3], &mut output),
        Err(pcu_facade::PcuExecutionError::UnsupportedNumericalOptions(
            _
        ))
    ));
    assert_eq!(
        output.map(f64::to_bits),
        [3.0_f64, 3.0, 3.0, 91.0, 91.0].map(f64::to_bits)
    );
    pcu_facade::global::use_defaults().unwrap();
}

#[test]
#[ignore = "Requires actual exact binary resident/mixed F64 fault and ownership proof."]
fn resident_and_mixed_binary_fault_tails_affinity_and_retry() {
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
    let pool = pcu_facade::PcuMemoryPoolId(91);
    let left = backend
        .upload_buffer(pool, &[1.0_f64, 2.0, f64::NAN, 81.0, 82.0])
        .unwrap();
    let right = backend
        .upload_buffer(pool, &[2.0_f64, 0.0, 1.0, 83.0, 84.0])
        .unwrap();
    let mut output = backend.upload_buffer(pool, &[91.0_f64; 5]).unwrap();
    let mut prepared = div_prepare_device::<3, _>(&backend).unwrap();
    assert!(
        matches!(prepared(&left, &right, &mut output), Err(fusion_pcu_metal::MetalOwnedDispatchError::Metal(MetalError::Arithmetic(fault))) if fault.invocation_id == 1 && fault.kind == PcuExecutionFaultKind::DivideByZero)
    );
    let mut actual = [0.0_f64; 5];
    backend.download_buffer(pool, &output, &mut actual).unwrap();
    assert_eq!(
        actual[3..]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [91.0_f64.to_bits(); 2]
    );
    let left = backend
        .upload_buffer(pool, &[1.0_f64, 2.0, 3.0, 81.0, 82.0])
        .unwrap();
    let right = backend
        .upload_buffer(pool, &[2.0_f64, 4.0, 6.0, 83.0, 84.0])
        .unwrap();
    prepared(&left, &right, &mut output).unwrap();
    backend.download_buffer(pool, &output, &mut actual).unwrap();
    assert_eq!(
        actual.map(f64::to_bits),
        [0.5_f64, 0.5, 0.5, 91.0, 91.0].map(f64::to_bits)
    );
    let short = pcu_facade::PcuDeviceBuffer::new(left.into_resource(), 2);
    assert!(matches!(
        prepared(&short, &right, &mut output),
        Err(fusion_pcu_metal::MetalOwnedDispatchError::Binding(
            pcu_facade::PcuOwnedDispatchBindingError::BufferTooSmall { .. }
        ))
    ));
    let resident = pcu_facade::PcuTensor::from_device_buffer(backend, right, &[5]).unwrap();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    div::<3>(&resident, &[2.0, 4.0, 6.0], &mut actual).unwrap();
    assert_eq!(
        actual.map(f64::to_bits),
        [1.0_f64, 1.0, 1.0, 91.0, 91.0].map(f64::to_bits)
    );
    let before = actual.map(f64::to_bits);
    assert!(
        matches!(div::<3>(&resident, &[2.0, 0.0, f64::NAN], &mut actual), Err(pcu_facade::PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id == 1 && fault.kind == PcuExecutionFaultKind::DivideByZero)
    );
    assert_eq!(actual.map(f64::to_bits), before);
    div::<3>(&resident, &[2.0, 4.0, 6.0], &mut actual).unwrap();
    assert_eq!(actual.map(f64::to_bits), before);
    pcu_facade::global::use_defaults().unwrap();
}

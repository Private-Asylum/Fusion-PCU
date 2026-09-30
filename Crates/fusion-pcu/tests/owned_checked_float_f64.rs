//! Strict checked binary64 source-to-device acceptance.
#![cfg(all(feature = "rocm", feature = "tensor"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuCheckedFloatConversion,
    PcuTensor,
};

#[pcu]
fn add(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[pcu(invocations: N)]
fn checked_cast<const N: usize>(input: &[f64; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}

#[pcu(invocations: N, flag(allow_gradual_underflow))]
fn checked_cast_gradual<const N: usize>(input: &[f64; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}

#[pcu(invocations: N, flag(reject_subnormal_result))]
fn checked_cast_strict<const N: usize>(input: &[f64; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}
#[pcu]
fn sub(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(lhs - rhs)
}
#[pcu]
fn mul(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(lhs * rhs)
}
#[pcu]
fn discarded(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    let _unused = lhs + rhs;
    pcu::identity(lhs)
}

fn fault(result: Result<PcuTensor<f64>, PcuExecutionError>, kind: PcuExecutionFaultKind) {
    match result {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, kind);
            assert_eq!(record.invocation_id, 1);
        }
        other => panic!("expected terminal {kind:?}, got {other:?}"),
    }
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn owned_binary64_checked_arithmetic_preserves_faults_and_results() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let sum = add(&[1.0, 2.0], &[2.0, 3.0]).unwrap();
    let difference = sub(&[3.0, 4.0], &[1.0, 2.0]).unwrap();
    let product = mul(&[2.0, 3.0], &[4.0, 5.0]).unwrap();
    let mut output = [0.0; 2];
    sum.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [3.0_f64.to_bits(), 5.0_f64.to_bits()]
    );
    difference.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [2.0_f64.to_bits(), 2.0_f64.to_bits()]
    );
    product.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [8.0_f64.to_bits(), 15.0_f64.to_bits()]
    );

    fault(
        add(&[1.0, f64::MAX], &[1.0, f64::MAX]),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    fault(
        discarded(&[1.0, f64::MAX], &[1.0, f64::MAX]),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    fault(
        sub(&[1.0, -f64::MAX], &[1.0, f64::MAX]),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    fault(
        mul(&[1.0, f64::MAX], &[1.0, 2.0]),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    fault(
        add(&[1.0, f64::INFINITY], &[1.0, 0.0]),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    fault(
        mul(&[1.0, f64::from_bits(1)], &[1.0, 0.5]),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    fault(
        mul(
            &[1.0, f64::MIN_POSITIVE],
            &[1.0, f64::from_bits(1.0_f64.to_bits() - 1)],
        ),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );

    let exact_subnormal = mul(&[f64::MIN_POSITIVE], &[0.5]).unwrap();
    exact_subnormal.read_into(&mut output[..1]).unwrap();
    assert_eq!(output[0].to_bits(), (1_u64 << 51));
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
#[allow(clippy::too_many_lines)] // Keep precision boundaries, random parity, fault IDs, and retry in one device fixture.
fn owned_f64_to_f32_checked_cast_matches_oracle_faults_and_retry() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let inputs = [1.0_f64, f64::from_bits(0x3ff0_0000_0000_0001), 3.0_f64];
    let mut output = [0.0_f32; 3];
    checked_cast(&inputs, &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        inputs.map(|x| x.pcu_checked_to_f32().unwrap().to_bits())
    );

    let pow2 = |exponent: i32| {
        let field = u64::try_from(exponent + 1023).expect("requested power is normal in f64");
        f64::from_bits(field << 52)
    };
    let min_normal = f64::from(f32::MIN_POSITIVE);
    let midpoint = min_normal - pow2(-150);
    let just_above_midpoint = min_normal - pow2(-151);
    let exact_min_subnormal = pow2(-149);
    let signed_zero = f64::from_bits(1_u64 << 63);
    let boundary_inputs = [
        midpoint,
        just_above_midpoint,
        signed_zero,
        exact_min_subnormal,
    ];
    let mut boundary_output = [0.0_f32; 4];
    checked_cast_gradual(&boundary_inputs, &mut boundary_output).unwrap();
    assert_eq!(
        boundary_output.map(f32::to_bits),
        boundary_inputs.map(|x| {
            x.pcu_checked_to_f32_with_policy(
                fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            )
            .unwrap()
            .to_bits()
        })
    );
    assert_eq!(boundary_output[2].to_bits(), 1_u32 << 31);
    assert_eq!(boundary_output[3].to_bits(), 1);

    // The exact half-ULP value just below minimum normal is tiny and inexact under the core
    // precision-rounding classification, even though gradual packing produces minimum normal.
    // Subtracting only 2^-151 places the value above that midpoint and default rounding succeeds.
    let mut normal_boundary = [0.0_f32; 2];
    checked_cast(
        &[just_above_midpoint, exact_min_subnormal],
        &mut normal_boundary,
    )
    .unwrap();
    assert_eq!(
        normal_boundary.map(f32::to_bits),
        [
            just_above_midpoint.pcu_checked_to_f32().unwrap().to_bits(),
            exact_min_subnormal.pcu_checked_to_f32().unwrap().to_bits(),
        ]
    );
    assert_eq!(normal_boundary[0].to_bits(), f32::MIN_POSITIVE.to_bits());
    for strict_input in [midpoint, exact_min_subnormal] {
        let mut one_output = [0.0_f32; 1];
        match checked_cast_strict(&[strict_input], &mut one_output) {
            Err(PcuExecutionError::ArithmeticFault(record)) => {
                assert_eq!(record.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
                assert_eq!(record.invocation_id, 0);
            }
            other => panic!("expected strict subnormal rejection, got {other:?}"),
        }
    }
    match checked_cast(&[midpoint], &mut [0.0_f32; 1]) {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
            assert_eq!(record.invocation_id, 0);
        }
        other => panic!("expected default tiny/inexact fault, got {other:?}"),
    }

    // Check adjacent binary64 values around the exact overflow midpoint between max-finite
    // and infinity. The immediately lower value rounds to max-finite; midpoint and above fault.
    let overflow_midpoint = f64::from(f32::MAX) + pow2(103);
    let overflow_below = f64::from_bits(overflow_midpoint.to_bits() - 1);
    let overflow_above = f64::from_bits(overflow_midpoint.to_bits() + 1);
    let mut max_output = [0.0_f32; 1];
    checked_cast(&[overflow_below], &mut max_output).unwrap();
    assert_eq!(max_output[0].to_bits(), f32::MAX.to_bits());
    for overflow in [overflow_midpoint, overflow_above] {
        assert_eq!(
            overflow.pcu_checked_to_f32().unwrap_err(),
            PcuExecutionFaultKind::ArithmeticOverflow
        );
        match checked_cast(&[overflow], &mut max_output) {
            Err(PcuExecutionError::ArithmeticFault(record)) => {
                assert_eq!(record.kind, PcuExecutionFaultKind::ArithmeticOverflow);
                assert_eq!(record.invocation_id, 0);
            }
            other => panic!("expected overflow midpoint fault, got {other:?}"),
        }
    }

    // Deterministic broad finite-bit parity corpus under gradual underflow. Filter only values
    // whose core oracle reports an overflow; every admitted result is compared by raw bits.
    let mut random_state = 0x6a09_e667_f3bc_c909_u64;
    let mut random_inputs = [0.0_f64; 512];
    for value in &mut random_inputs {
        loop {
            random_state ^= random_state << 13;
            random_state ^= random_state >> 7;
            random_state ^= random_state << 17;
            let exponent = (random_state >> 40) % 0x7ff;
            let bits = (random_state & 0x800f_ffff_ffff_ffff) | (exponent << 52);
            let candidate = f64::from_bits(bits);
            if candidate
                .pcu_checked_to_f32_with_policy(
                    fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                )
                .is_ok()
            {
                *value = candidate;
                break;
            }
        }
    }
    let mut random_output = [0.0_f32; 512];
    checked_cast_gradual(&random_inputs, &mut random_output).unwrap();
    assert_eq!(
        random_output.map(f32::to_bits),
        random_inputs.map(|x| {
            x.pcu_checked_to_f32_with_policy(
                fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            )
            .unwrap()
            .to_bits()
        })
    );

    for (bad, kind) in [
        (f64::INFINITY, PcuExecutionFaultKind::InvalidFloatingOperand),
        (f64::NAN, PcuExecutionFaultKind::InvalidFloatingOperand),
        (f64::MAX, PcuExecutionFaultKind::ArithmeticOverflow),
        (pow2(-150), PcuExecutionFaultKind::ArithmeticUnderflow),
    ] {
        let input = [1.0_f64, bad];
        let mut out = [0.0_f32; 2];
        match checked_cast(&input, &mut out) {
            Err(PcuExecutionError::ArithmeticFault(record)) => {
                assert_eq!(record.kind, kind);
                assert_eq!(record.invocation_id, 1);
            }
            other => panic!("expected checked conversion fault {kind:?}, got {other:?}"),
        }
        checked_cast(&[2.0_f64, 3.0], &mut out).unwrap();
        assert_eq!(
            out.map(f32::to_bits),
            [2.0_f32.to_bits(), 3.0_f32.to_bits()]
        );
    }
    global::clear_thread_cache().unwrap();
}

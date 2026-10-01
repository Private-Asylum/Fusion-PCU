//! Actual source optimizer execution, precision freedoms and escaped resident ownership.
#![cfg(all(feature = "rocm", feature = "tensor"))]

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuTensor,
};

#[pcu(flag(non_strict), flag(native_compound))]
fn update<const N: usize>(
    weights: &[f32; N],
    gradient: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}

#[pcu]
fn identity<const N: usize>(input: &[f32; N]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(flag(native_compound), flag(preserve_precision))]
fn preserved_witness(
    weights: &[f32; 17],
    gradient: &[f32; 17],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 1.000_000_1_f32)
}

#[pcu(flag(native_compound), flag(backend_precision))]
fn contracted_witness(
    weights: &[f32; 17],
    gradient: &[f32; 17],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 1.000_000_1_f32)
}

#[pcu(flag(native_compound), flag(backend_precision))]
fn parent_with_preserved_helper(
    weights: &[f32; 17],
    gradient: &[f32; 17],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    preserved_witness(weights, gradient)
}

#[pcu(flag(native_compound))]
fn project_update(
    weights: &[[f32; 2]; 2],
    left: &[[f32; 2]; 2],
    right: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let gradient = pcu::matmul(left, right)?;
    pcu::sgd_update(weights, &gradient, 0.5_f32)
}

#[pcu]
fn checked(weights: &[f32; 1], gradient: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}

#[pcu(flag(strict), flag(native_compound))]
fn strict(weights: &[f32; 1], gradient: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}

#[pcu(flag(native_compound), flag(deterministic))]
fn portable(weights: &[f32; 1], gradient: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}

#[pcu(flag(native_compound), flag(reject_subnormal_result))]
fn tight(weights: &[f32; 1], gradient: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}

#[pcu(flag(native_compound), flag(allow_gradual_underflow))]
fn gradual(weights: &[f32; 1], gradient: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}

fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn actual_optimizer_changes_inputs_composes_and_retains_escaped_output() {
    configure();
    for phase in 0..3_u16 {
        let weights: [f32; 65] =
            core::array::from_fn(|index| f32::from(u16::try_from(index).unwrap() + phase));
        let gradient: [f32; 65] =
            core::array::from_fn(|index| f32::from(i16::try_from(index % 5).unwrap() - 2));
        let expected = core::array::from_fn::<_, 65, _>(|index| {
            weights[index]
                .pcu_checked_sub(0.5_f32.pcu_checked_mul(gradient[index]).unwrap())
                .unwrap()
        });
        let first = update(&weights, &gradient).unwrap();
        let weights_owner = identity(&weights).unwrap();
        let gradient_owner = identity(&gradient).unwrap();
        let second = update::<65>(&weights_owner, &gradient_owner).unwrap();
        drop(weights_owner);
        drop(gradient_owner);
        global::clear_thread_cache().unwrap();
        let mut observed = [99.0_f32; 66];
        for result in [first, second] {
            result.read_into(&mut observed[..65]).unwrap();
            assert_eq!(result.shape(), &[65]);
            for (actual, wanted) in observed.iter().zip(expected) {
                assert_eq!(actual.to_bits(), wanted.to_bits());
            }
            assert_eq!(observed[65].to_bits(), 99.0_f32.to_bits());
        }
    }
    let result = project_update(
        &[[1.0, 2.0], [3.0, 4.0]],
        &[[1.0, 2.0], [3.0, 4.0]],
        &[[2.0, 1.0], [0.0, 2.0]],
    )
    .unwrap();
    let mut observed = [0.0; 4];
    result.read_into(&mut observed).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(
        observed.map(f32::to_bits),
        [0.0_f32, -0.5, 0.0, -1.5].map(f32::to_bits)
    );
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn explicit_precision_and_helper_override_choose_distinct_rounding() {
    configure();
    // rate=(1+2^-23), gradient=(1-2^-23). Separately rounded multiplication becomes1,
    // so subtraction yields+0; a fused multiply/subtract retains the exact residual2^-46.
    let weights = [1.0; 17];
    let gradient = [f32::from_bits(0x3f7f_fffe); 17];
    let mut observed = [0.0; 17];
    preserved_witness(&weights, &gradient)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert!(observed.iter().all(|value| value.to_bits() == 0));
    contracted_witness(&weights, &gradient)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert!(
        observed
            .iter()
            .all(|value| value.to_bits() == (127 - 46) << 23)
    );
    parent_with_preserved_helper(&weights, &gradient)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert!(observed.iter().all(|value| value.to_bits() == 0));
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn native_permission_keeps_unsupported_profiles_and_operational_failures_explicit() {
    configure();
    for error in [
        checked(&[1.0], &[0.0]).unwrap_err(),
        strict(&[1.0], &[0.0]).unwrap_err(),
        portable(&[1.0], &[0.0]).unwrap_err(),
        tight(&[1.0], &[0.0]).unwrap_err(),
        gradual(&[1.0], &[0.0]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"), "{error:?}");
    }
    assert!(update::<0>(&[], &[]).is_err());
    let mut observed = [0.0];
    update(&[f32::MAX], &[-f32::MAX])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert!(observed[0].is_infinite());
    update(&[1.0], &[f32::NAN])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert!(observed[0].is_nan());
    update(&[3.0], &[1.0])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(observed[0].to_bits(), 2.5_f32.to_bits());
    global::clear_thread_cache().unwrap();
}

//! Actual owned-source native loss, with explicit admission and independent exact-domain oracle.
#![cfg(all(feature = "rocm", feature = "tensor"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu(flag(non_strict), flag(native_compound))]
fn native<const N: usize>(
    prediction: &[f32; N],
    target: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(non_strict), flag(native_compound), flag(backend_precision))]
fn optimized<const N: usize>(
    prediction: &[f32; N],
    target: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu]
fn identity<const N: usize>(input: &[f32; N]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn checked(prediction: &[f32; 1], target: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(strict), flag(native_compound))]
fn strict(prediction: &[f32; 1], target: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(native_compound), flag(deterministic))]
fn portable(prediction: &[f32; 1], target: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(native_compound), flag(reject_subnormal_result))]
fn tight(prediction: &[f32; 1], target: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(native_compound))]
fn unsupported_width(
    prediction: &[f64; 1],
    target: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(native_compound))]
fn project_loss(
    left: &[[f32; 2]; 2],
    right: &[[f32; 2]; 2],
    target: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let projected = pcu::matmul(left, right)?;
    pcu::mean_squared_error(&projected, target)
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
fn native_loss_changes_inputs_composes_and_retains_escaped_scalar() {
    configure();
    for phase in 0..3_u16 {
        let prediction: [f32; 65] = core::array::from_fn(|index| {
            f32::from(u16::try_from(index % 9).unwrap()) + f32::from(phase)
        });
        let target = [2.0_f32; 65];
        // Every square and sum is an exact small integer. Only the declared reciprocal/count
        // scaling rounds; this oracle does not assume general rocBLAS reduction bit identity.
        let expected = prediction
            .iter()
            .map(|value| (value - 2.0).powi(2))
            .sum::<f32>()
            * (1.0 / 65.0);
        let output = native(&prediction, &target).unwrap();
        assert_eq!(output.shape(), &[]);
        let mut observed = [0.0_f32];
        output.read_into(&mut observed).unwrap();
        assert_eq!(observed[0].to_bits(), expected.to_bits());

        let prediction_owner = identity(&prediction).unwrap();
        let target_owner = identity(&target).unwrap();
        let escaped = optimized::<65>(&prediction_owner, &target_owner).unwrap();
        drop(prediction_owner);
        drop(target_owner);
        global::clear_thread_cache().unwrap();
        escaped.read_into(&mut observed).unwrap();
        assert_eq!(observed[0].to_bits(), expected.to_bits());
    }
    let output = project_loss(
        &[[1.0, 2.0], [3.0, 4.0]],
        &[[2.0, 1.0], [0.0, 2.0]],
        &[[1.0, 2.0], [3.0, 4.0]],
    )
    .unwrap();
    let mut observed = [0.0_f32];
    output.read_into(&mut observed).unwrap();
    // Product [[2,5],[6,11]], squared residuals 1+9+9+49, mean=17.
    assert_eq!(observed[0].to_bits(), 17.0_f32.to_bits());
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn native_loss_permissions_do_not_weaken_default_or_operational_errors() {
    configure();
    for error in [
        checked(&[1.0], &[0.0]).unwrap_err(),
        strict(&[1.0], &[0.0]).unwrap_err(),
        portable(&[1.0], &[0.0]).unwrap_err(),
        tight(&[1.0], &[0.0]).unwrap_err(),
        unsupported_width(&[1.0], &[0.0]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"), "{error:?}");
    }
    assert!(native::<0>(&[], &[]).is_err());
    let mut observed = [0.0_f32];
    native(&[f32::MAX], &[-f32::MAX])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert!(observed[0].is_infinite());
    native(&[f32::NAN], &[0.0])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert!(observed[0].is_nan());
    native(&[f32::from_bits(1)], &[0.0])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(observed[0].to_bits(), 0.0_f32.to_bits());
    // Rejected and exceptional profiles cannot poison a subsequent ordinary finite call.
    native(&[3.0], &[1.0])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(observed[0].to_bits(), 4.0_f32.to_bits());
    global::clear_thread_cache().unwrap();
}

//! Actual generic checked float source specialization keeps backend width admission explicit.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalSession,
};

#[pcu(invocations = N, flag(strict), crate_path = ::pcu_facade)]
fn product<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}

#[test]
#[ignore = "Requires actual generic F32/F64 source on Metal."]
fn generic_source_specializes_exact_float_width() {
    let _policy_guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let mut single = product_prepare::<f32, 3, _>(&session).unwrap();
    let mut double = product_prepare::<f64, 3, _>(&session).unwrap();
    let mut f32_output = [91.0_f32; 5];
    let mut f64_output = [91.0_f64; 5];
    single(&[1.0, -2.0, 3.0], &[2.0, 3.0, 4.0], &mut f32_output).unwrap();
    double(&[1.0, -2.0, 3.0], &[2.0, 3.0, 4.0], &mut f64_output).unwrap();
    assert_eq!(
        f32_output.map(f32::to_bits),
        [2.0_f32, -6.0, 12.0, 91.0, 91.0].map(f32::to_bits)
    );
    assert_eq!(
        f64_output.map(f64::to_bits),
        [2.0_f64, -6.0, 12.0, 91.0, 91.0].map(f64::to_bits)
    );
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    product::<f32, 3>(&[2.0, 3.0, 4.0], &[4.0, 5.0, 6.0], &mut f32_output).unwrap();
    product::<f64, 3>(&[2.0, 3.0, 4.0], &[4.0, 5.0, 6.0], &mut f64_output).unwrap();
    assert_eq!(
        f32_output.map(f32::to_bits),
        [8.0_f32, 15.0, 24.0, 91.0, 91.0].map(f32::to_bits)
    );
    assert_eq!(
        f64_output.map(f64::to_bits),
        [8.0_f64, 15.0, 24.0, 91.0, 91.0].map(f64::to_bits)
    );
    pcu_facade::global::use_defaults().unwrap();
}

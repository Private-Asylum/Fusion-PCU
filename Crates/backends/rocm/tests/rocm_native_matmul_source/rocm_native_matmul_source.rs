//! Actual annotated native compound source: bounded parity and admission separation.
#![cfg(feature = "tensor")]
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuScalar,
    PcuTensor,
};
#[path = "../../benches/native_matmul/oracle.rs"]
mod oracle;
use oracle::Scalar;

#[pcu(flag(non_strict), flag(native_compound))]
fn native<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}
#[pcu(flag(non_strict), flag(native_compound), flag(backend_precision))]
fn optimized<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}
#[pcu(flag(strict), flag(native_compound))]
fn incompatible_strict(
    left: &[[f32; 1]; 1],
    right: &[[f32; 1]; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}
#[pcu(flag(non_strict), flag(native_compound), flag(deterministic))]
fn incompatible_portable(
    left: &[[f32; 1]; 1],
    right: &[[f32; 1]; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}
#[pcu(flag(non_strict), flag(checked_compound))]
fn checked_boundary(
    left: &[[f32; 1]; 1],
    right: &[[f32; 1]; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}
#[pcu(flag(non_strict), flag(native_compound), flag(reject_subnormal_result))]
fn incompatible_underflow(
    left: &[[f32; 1]; 1],
    right: &[[f32; 1]; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}
#[pcu]
fn identity<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        block_size: 256,
        numerical_mode: PcuNumericalMode::Boundary,
        float_underflow: PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ..Default::default()
    })
    .unwrap();
}
fn check<T: Scalar, const R: usize, const K: usize, const C: usize>() {
    eprintln!("native source {} {R}x{K}x{C}", T::LABEL);
    let mut observed = vec![T::default(); R * C];
    let phases = if K > 256 { [0, 1, 2] } else { [0, 1, 17] };
    for phase in phases {
        let (left, right) = oracle::fill::<T, R, K, C>(phase);
        let wanted = oracle::expected::<T, R, K, C>(phase);
        let output = native::<T, R, K, C>(&left, &right).unwrap();
        output.read_into(&mut observed).unwrap();
        oracle::verify(&wanted, &observed);
        let left_owner = identity::<T, R, K>(&left).unwrap();
        let right_owner = identity::<T, K, C>(&right).unwrap();
        let escaped = native::<T, R, K, C>(&left_owner, &right_owner).unwrap();
        let optimized = optimized::<T, R, K, C>(&left_owner, &right_owner).unwrap();
        drop(left_owner);
        drop(right_owner);
        drop(left);
        drop(right);
        global::clear_thread_cache().unwrap();
        escaped.read_into(&mut observed).unwrap();
        oracle::verify(&wanted, &observed);
        optimized.read_into(&mut observed).unwrap();
        oracle::verify(&wanted, &observed);
    }
}
#[test]
fn bounded_integer_oracle_changes_inputs_and_preserves_f32_f64_values() {
    for phase in [0, 1, 17, 34] {
        let f32_expected = oracle::expected::<f32, 64, 256, 64>(phase);
        let f64_expected = oracle::expected::<f64, 64, 256, 64>(phase);
        assert!(
            f32_expected
                .iter()
                .all(|&value| value > 0.0 && value <= 8960.0)
        );
        for (&narrow, &wide) in f32_expected.iter().zip(&f64_expected) {
            assert_eq!(f64::from(narrow).to_bits(), wide.to_bits());
        }
    }
    assert_ne!(
        oracle::expected::<f32, 4, 4, 4>(0),
        oracle::expected::<f32, 4, 4, 4>(1)
    );
}
#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn actual_native_source_parity_escaped_owners_and_requested_precision() {
    configure();
    global::clear_thread_cache().unwrap();
    check::<f32, 4, 4, 4>();
    check::<f64, 4, 4, 4>();
    check::<f32, 32, 32, 32>();
    check::<f64, 32, 32, 32>();
    check::<f32, 64, 256, 64>();
    check::<f64, 64, 256, 64>();
    global::clear_thread_cache().unwrap();
}
#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn source_native_admission_is_separate_from_checked_faults_and_retries() {
    configure();
    let left = [[2.0_f32]];
    let right = [[3.0_f32]];
    for error in [
        incompatible_strict(&left, &right).unwrap_err(),
        incompatible_portable(&left, &right).unwrap_err(),
        checked_boundary(&left, &right).unwrap_err(),
        incompatible_underflow(&left, &right).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"), "{error:?}");
    }
    // BackendDefined does not promise numerical fault diagnostics. These single-term
    // controls assert result classes only, without NaN payload or subnormal guarantees.
    for (operand, infinite) in [(f32::MAX, true), (f32::NAN, false)] {
        let output = native::<f32, 1, 1, 1>(&[[operand]], &[[2.0]]).unwrap();
        let mut values = [0.0_f32];
        output.read_into(&mut values).unwrap();
        if infinite {
            assert!(values[0].is_infinite());
        } else {
            assert!(values[0].is_nan());
        }
    }
    let underflow = native::<f32, 1, 1, 1>(&[[f32::from_bits(1)]], &[[0.5]]).unwrap();
    let mut rounded = [1.0_f32];
    underflow.read_into(&mut rounded).unwrap();
    assert_eq!(rounded[0].to_bits(), 0.0_f32.to_bits());
    let underflow = native::<f64, 1, 1, 1>(&[[f64::from_bits(1)]], &[[0.5]]).unwrap();
    let mut rounded = [1.0_f64];
    underflow.read_into(&mut rounded).unwrap();
    assert_eq!(rounded[0].to_bits(), 0.0_f64.to_bits());
    for (operand, infinite) in [(f64::MAX, true), (f64::NAN, false)] {
        let output = native::<f64, 1, 1, 1>(&[[operand]], &[[2.0]]).unwrap();
        let mut values = [0.0_f64];
        output.read_into(&mut values).unwrap();
        if infinite {
            assert!(values[0].is_infinite());
        } else {
            assert!(values[0].is_nan());
        }
    }
    let output = native::<f32, 1, 1, 1>(&left, &right).unwrap();
    let mut values = [0.0_f32];
    output.read_into(&mut values).unwrap();
    assert_eq!(values[0].to_bits(), 6.0_f32.to_bits());
    global::clear_thread_cache().unwrap();
}

#[test]
fn heavy_closed_form_oracle_matches_an_independent_direct_dot_and_heap_shape() {
    assert_eq!(oracle::phase_count::<2048>(), 3);
    assert_eq!(oracle::phase_count::<256>(), 35);
    // The same large-K branch at small row/column counts lets a plain integer dot audit
    // every closed-form output; nonsquare axes and three phases detect swapped dimensions.
    for phase in 0..3 {
        let (left, right) = oracle::fill::<f64, 3, 2048, 5>(phase);
        let wanted = oracle::expected::<f64, 3, 2048, 5>(phase);
        for row in 0..3 {
            for column in 0..5 {
                let sum: f64 = (0..2048)
                    .map(|depth| left[row][depth] * right[depth][column])
                    .sum();
                assert_eq!(sum.to_bits(), wanted[row * 5 + column].to_bits());
            }
        }
        let narrow = oracle::expected::<f32, 1024, 2048, 1024>(phase);
        let wide = oracle::expected::<f64, 1024, 2048, 1024>(phase);
        assert_eq!(narrow.len(), 1024 * 1024);
        assert!(
            narrow
                .iter()
                .all(|&value| value > 0.0 && value < f32::from_bits(0x4B80_0000))
        );
        for (&narrow, &wide) in narrow.iter().zip(&wide) {
            assert_eq!(f64::from(narrow).to_bits(), wide.to_bits());
        }
        let (left, right) = oracle::fill::<f32, 1024, 2048, 1024>(phase);
        assert_eq!(left.len(), 1024);
        assert_eq!(right.len(), 2048);
    }
}
#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn workload_heavy_native_source_parity_and_escaped_owners() {
    configure();
    global::clear_thread_cache().unwrap();
    check::<f32, 1024, 2048, 1024>();
    check::<f64, 1024, 2048, 1024>();
    global::clear_thread_cache().unwrap();
}

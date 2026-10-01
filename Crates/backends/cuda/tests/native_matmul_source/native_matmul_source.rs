//! Actual authored `MatMul` uses retained Lt plans, changing inputs and terminal fresh outputs.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/native_matmul/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuScalar,
};
fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        block_size: 256,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
fn exercise<T: PcuScalar + PartialEq + std::fmt::Debug>(
    left: [[T; 3]; 2],
    right: [[T; 2]; 3],
    expected: [T; 4],
) {
    for optimized in [false, true] {
        global::clear_thread_cache().unwrap();
        #[cfg(feature = "allocation-census")]
        fusion_pcu_cuda::reset_cublaslt_api_census();
        for iteration in 0..3 {
            let output = source::product(&left, &right, optimized).unwrap();
            let mut actual = expected;
            output.read_into(&mut actual).unwrap();
            assert_eq!(actual, expected);
            #[cfg(feature = "allocation-census")]
            {
                let census = fusion_pcu_cuda::cublaslt_api_census();
                assert_eq!(census.loads, 1, "actual source prepares one exact Lt plan");
                assert_eq!(
                    census.heuristics, 1,
                    "warm source execution never reruns heuristics"
                );
                assert_eq!(census.matmuls, iteration + 1);
            }
            #[cfg(not(feature = "allocation-census"))]
            let _ = iteration;
        }
        let escaped = source::product(&left, &right, optimized).unwrap();
        global::clear_thread_cache().unwrap();
        let mut actual = expected;
        escaped.read_into(&mut actual).unwrap();
        assert_eq!(actual, expected);
    }
}
#[test]
#[ignore = "requires authorized idle CUDA hardware and stable cuBLASLt; run serially"]
fn actual_source_lt_cold_once_warm_algorithm_retained_and_output_survives_cache() {
    configure();
    for phase in 0..3 {
        let value = f32::from(u16::try_from(phase + 1).unwrap());
        exercise(
            [[value, 2.0_f32, 3.0], [4.0, 5.0, 6.0]],
            [[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [value + 21.0, value + value + 26.0, 49.0, 64.0],
        );
        let value = f64::from(phase + 1);
        exercise(
            [[value, 2.0_f64, 3.0], [4.0, 5.0, 6.0]],
            [[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [value + 21.0, value + value + 26.0, 49.0, 64.0],
        );
    }
    global::clear_thread_cache().unwrap();
}
#[test]
#[ignore = "requires authorized idle CUDA hardware and stable cuBLASLt; run serially"]
fn actual_source_preserve_precision_nonfinite_permission_finite_retry_and_cold_negatives() {
    configure();
    let left = [[1.000_000_1_f32, 1.0]];
    let right = [[1.0_f32], [-1.0]];
    let mut actual = [0.0_f32];
    source::preserve(&left, &right)
        .unwrap()
        .read_into(&mut actual)
        .unwrap();
    assert_eq!(actual[0].to_bits(), (1.000_000_1_f32 - 1.0).to_bits());
    let left = [[1.000_000_000_000_000_2_f64, 1.0]];
    let right = [[1.0_f64], [-1.0]];
    let mut actual = [0.0_f64];
    source::preserve(&left, &right)
        .unwrap()
        .read_into(&mut actual)
        .unwrap();
    assert_eq!(
        actual[0].to_bits(),
        (1.000_000_000_000_000_2_f64 - 1.0).to_bits()
    );
    for optimized in [false, true] {
        let mut nonfinite = [0.0_f32];
        source::product(&[[f32::INFINITY]], &[[1.0_f32]], optimized)
            .unwrap()
            .read_into(&mut nonfinite)
            .unwrap();
        assert!(nonfinite[0].is_infinite());
        let mut finite = [0.0_f32];
        source::product(&[[2.0_f32]], &[[3.0_f32]], optimized)
            .unwrap()
            .read_into(&mut finite)
            .unwrap();
        assert_eq!(finite[0].to_bits(), 6.0_f32.to_bits());
    }
    let left = [[1.0_f32]];
    let right = [[1.0_f32]];
    for error in [
        source::checked_boundary(&left, &right).unwrap_err(),
        source::strict_native(&left, &right).unwrap_err(),
        source::portable_native(&left, &right).unwrap_err(),
        source::tight_native(&left, &right).unwrap_err(),
        source::gradual_native(&left, &right).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"), "{error:?}");
    }
    assert!(source::preserve(&[[1_i32]], &[[1_i32]]).is_err());
    global::clear_thread_cache().unwrap();
}

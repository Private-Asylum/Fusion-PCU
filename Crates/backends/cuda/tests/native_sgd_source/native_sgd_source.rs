//! Actual authored CUDA optimizer calls: frozen rates, owned output and native precision offers.
extern crate pcu_facade as fusion_pcu;

#[path = "../../benches/native_sgd/source/source.rs"]
mod source;

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
};
use fusion_pcu::dialect::tensor::Graph;

#[pcu]
fn identity<const N: usize>(input: &[f32; N]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(flag(non_strict), flag(native_compound), flag(non_deterministic))]
fn positive_zero(
    weights: &[f32; 4],
    gradient: &[f32; 4],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.0_f32)
}
#[pcu(flag(non_strict), flag(native_compound), flag(non_deterministic))]
fn negative_zero(
    weights: &[f32; 4],
    gradient: &[f32; 4],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, -0.0_f32)
}
#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(backend_precision),
    flag(non_deterministic)
)]
fn optimized_positive_zero(
    weights: &[f32; 4],
    gradient: &[f32; 4],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.0_f32)
}
#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(backend_precision),
    flag(non_deterministic)
)]
fn optimized_negative_zero(
    weights: &[f32; 4],
    gradient: &[f32; 4],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, -0.0_f32)
}
#[pcu(flag(non_strict), flag(native_compound), flag(non_deterministic))]
fn extreme_rate(
    weights: &[f32; 1],
    gradient: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 3.402_823_5e38_f32)
}
#[pcu(flag(non_strict), flag(native_compound), flag(non_deterministic))]
fn unsupported_width(
    weights: &[f64; 1],
    gradient: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.25_f32)
}

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

fn verify(output: &PcuTensor<f32>, expected: &[f32]) {
    assert_eq!(output.shape(), [expected.len()]);
    let mut actual = vec![0.0; expected.len()];
    output.read_into(&mut actual).unwrap();
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "source SGD lane{index}"
        );
    }
}

#[test]
fn typed_graph_rejects_nonfinite_rates_dtype_and_shape_without_a_device() {
    let mut graph = Graph::default();
    let w = graph.input_typed::<f32>([17]).unwrap();
    let g = graph.input_typed::<f32>([17]).unwrap();
    let wrong_shape = graph.input_typed::<f32>([18]).unwrap();
    let wrong_type = graph.input_typed::<f64>([17]).unwrap();
    assert!(graph.sgd_update_typed::<f32>(w, wrong_shape, 0.25).is_err());
    assert!(
        graph
            .sgd_update(w.erase(), wrong_type.erase(), 0.25)
            .is_err()
    );
    for rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(graph.sgd_update_typed::<f32>(w, g, rate).is_err());
    }
    for rate in [0.0_f32, -0.0, -0.25, f32::MAX, -f32::MAX] {
        assert!(graph.sgd_update_typed::<f32>(w, g, rate).is_ok());
    }
}

#[test]
#[ignore = "requires authorized idle CUDA hardware; run serially"]
fn authored_updates_change_both_inputs_and_keep_escaped_borrowed_resident_owners() {
    configure();
    for optimized in [false, true] {
        for phase in 0..3_i16 {
            let w: [f32; 17] = core::array::from_fn(|i| {
                f32::from(i16::try_from(i % 9).unwrap() - 4 + phase) / 8.0
            });
            let g: [f32; 17] = core::array::from_fn(|i| {
                f32::from(i16::try_from(i % 7).unwrap() - 3 - phase) / 8.0
            });
            let expected: [f32; 17] = core::array::from_fn(|i| {
                let wu = i16::try_from(i % 9).unwrap() - 4 + phase;
                let gu = i16::try_from(i % 7).unwrap() - 3 - phase;
                f32::from(4 * wu + gu) / 32.0
            });
            let output = source::update(&w, &g, optimized).unwrap();
            verify(&output, &expected);
            let w_owner = identity(&w).unwrap();
            let g_owner = identity(&g).unwrap();
            let escaped = if optimized {
                source::optimized::<17>(&w_owner, &g_owner).unwrap()
            } else {
                source::preserve::<17>(&w_owner, &g_owner).unwrap()
            };
            // Borrowing optimizer operands must leave both original owners readable.
            verify(&w_owner, &w);
            verify(&g_owner, &g);
            drop(w_owner);
            drop(g_owner);
            global::clear_thread_cache().unwrap();
            verify(&escaped, &expected);
        }
    }
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires authorized idle CUDA hardware; run serially"]
fn authored_rate_bits_signed_zero_cache_identity_fma_and_exception_retry() {
    configure();
    let w = [0.0, -0.0, 0.0, -0.0];
    let g = [1.0, 1.0, -1.0, -1.0];
    for output in [
        positive_zero(&w, &g).unwrap(),
        optimized_positive_zero(&w, &g).unwrap(),
    ] {
        verify(&output, &[0.0, -0.0, 0.0, 0.0]);
    }
    for output in [
        negative_zero(&w, &g).unwrap(),
        optimized_negative_zero(&w, &g).unwrap(),
    ] {
        verify(&output, &[0.0, 0.0, 0.0, -0.0]);
    }
    for _ in 0..3 {
        verify(&source::preserve(&[1.0], &[1.0]).unwrap(), &[1.25]);
        verify(&source::other_rate(&[1.0], &[1.0]).unwrap(), &[0.5]);
        verify(&source::preserve(&[1.0], &[1.0]).unwrap(), &[1.25]);
    }
    assert_eq!(1.000_000_1_f32.to_bits(), 0x3f80_0001);
    let g = [f32::from_bits(0x3f7f_fffe)];
    verify(&source::witness_preserve(&[1.0], &g).unwrap(), &[0.0]);
    verify(
        &source::witness_optimized(&[1.0], &g).unwrap(),
        &[f32::from_bits((127_u32 - 46) << 23)],
    );
    assert_eq!(3.402_823_5e38_f32.to_bits(), f32::MAX.to_bits());
    verify(&extreme_rate(&[1.0], &[0.0]).unwrap(), &[1.0]);
    for optimized in [false, true] {
        let output = source::update(&[f32::INFINITY, 1.0], &[1.0, f32::NAN], optimized).unwrap();
        let mut actual = [0.0; 2];
        output.read_into(&mut actual).unwrap();
        assert!(actual[0].is_infinite() && actual[1].is_nan());
        verify(&source::update(&[1.0], &[1.0], optimized).unwrap(), &[1.25]);
    }
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires authorized idle CUDA hardware; run serially"]
fn authored_cold_policy_dtype_and_resident_shape_rejections_preserve_owner_reads() {
    configure();
    for error in [
        source::checked(&[1.0], &[1.0]).unwrap_err(),
        source::strict(&[1.0], &[1.0]).unwrap_err(),
        source::portable(&[1.0], &[1.0]).unwrap_err(),
        source::tight(&[1.0], &[1.0]).unwrap_err(),
        unsupported_width(&[1.0], &[1.0]).unwrap_err(),
    ] {
        assert!(format!("{error:?}").contains("Unsupported"), "{error:?}");
    }
    assert!(source::preserve::<0>(&[], &[]).is_err());
    let w = identity(&[1.0, 2.0]).unwrap();
    let g = identity(&[1.0, 2.0, 3.0]).unwrap();
    assert!(source::preserve::<2>(&w, &g).is_err());
    verify(&w, &[1.0, 2.0]);
    verify(&g, &[1.0, 2.0, 3.0]);
    verify(&source::preserve(&[1.0], &[1.0]).unwrap(), &[1.25]);
    global::clear_thread_cache().unwrap();
}

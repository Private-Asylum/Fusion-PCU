//! Concrete typed helper composition on the selected device.
#![cfg(feature = "rocm")]

use fusion_pcu::pcu;

mod arithmetic {
    use fusion_pcu::pcu;

    #[pcu]
    fn multiply(value: f64, factor: f64) -> f64 {
        value * factor
    }

    #[pcu]
    pub fn affine(value: f64, factor: f64) -> f64 {
        multiply(value, factor) + 1.0
    }
}

use arithmetic::affine as aliased_affine;

#[pcu(invocations: N)]
fn transform<const N: usize>(seed: &f64, input: &[f64; N], output: &mut [f64; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = aliased_affine(input[id], *seed);
}

#[test]
fn aliased_helper_ir_is_admitted_by_the_double_precision_profile() {
    let bindings = transform_bindings();
    let builder = transform_ir::<4>(&bindings).expect("lower helper kernel");
    let kernel = builder.ir();
    assert!(
        fusion_pcu::validate_checked_float_map_kernel(
            &kernel,
            fusion_pcu::PcuValueType::f64(),
            fusion_pcu::PcuValueTypeCaps::FLOAT64
        )
        .is_ok(),
        "f64 helper IR fails its scalar profile: {kernel:?}"
    );
}

#[test]
#[ignore = "requires a working ROCm device"]
#[allow(clippy::suboptimal_flops)] // Match the kernel's separate multiply/add rounding contract.
fn aliased_f64_helpers_preserve_double_precision_and_fresh_seeds() {
    fusion_pcu::global::use_defaults().unwrap();
    let input = [
        0.1_f64,
        1.0e40,
        f64::from_bits(0x3ff0_0000_0000_0001),
        -1.0e40,
    ];
    let mut output = [0.0_f64; 4];
    for seed in [1.000_000_000_000_000_2_f64, -0.5, 2.0] {
        transform(&seed, &input, &mut output).unwrap();
        let expected = input.map(|value| (value * seed + 1.0).to_bits());
        assert_eq!(output.map(f64::to_bits), expected);
    }
    fusion_pcu::global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device"]
#[allow(clippy::suboptimal_flops)] // The witness must preserve separate Mul/Add rounding.
fn f64_helpers_preserve_separate_rounding_against_an_fma_witness() {
    fusion_pcu::global::use_defaults().unwrap();
    let value = f64::from_bits(0xbff0_0000_0200_0000); // -(1 + 2^-27)
    let seed = f64::from_bits(0x3fef_ffff_fc00_0000); // 1 - 2^-27
    let expected = value * seed + 1.0;
    assert_eq!(expected.to_bits(), 0.0_f64.to_bits());
    assert_ne!(value.mul_add(seed, 1.0).to_bits(), expected.to_bits());
    let mut output = [99.0_f64; 1];
    transform(&seed, &[value], &mut output).unwrap();
    assert_eq!(output[0].to_bits(), expected.to_bits());
    fusion_pcu::global::clear_thread_cache().unwrap();
}

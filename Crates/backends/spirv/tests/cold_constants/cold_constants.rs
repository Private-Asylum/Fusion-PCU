//! Whole emitted modules must retain exact metadata under host rounding/flush controls.
#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
// The production-generated module retains its parent visibility in this test-only import.
#[allow(clippy::redundant_pub_crate)]
#[path = "../../checked_compound/bytecode/bytecode.rs"]
mod bytecode;
#[path = "../../../cpu/tests/cold_constants/environment/environment.rs"]
mod environment;
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuScalarType,
    PcuFloatUnderflowPolicy,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_checked_compound_to_spirv,
    PcuSpirvCompoundOperation,
    PcuSpirvLoweringOptions,
    PcuSpirvError,
};
use std::hint::black_box;

fn golden(scalar: PcuScalarType, values: [u32; 9]) -> Vec<u32> {
    let (source, offsets) = if scalar == PcuScalarType::F32 {
        (bytecode::F32_WORDS, bytecode::F32_OFFSETS.as_slice())
    } else {
        (bytecode::F64_WORDS, bytecode::F64_OFFSETS.as_slice())
    };
    let mut words = source.to_vec();
    let options = PcuSpirvLoweringOptions::default();
    words[1] = options.version.0;
    words[2] = options.generator;
    for (&offset, value) in offsets.iter().zip(values) {
        words[offset] = value;
    }
    words
}
fn compare(
    scalar: PcuScalarType,
    operation: PcuSpirvCompoundOperation,
    values: [u32; 9],
    policy: PcuFloatUnderflowPolicy,
    failures: &mut usize,
) {
    let expected = golden(scalar, values);
    let mut actual = Vec::new();
    let (_, profile) = lower_checked_compound_to_spirv(
        scalar,
        operation,
        policy,
        PcuSpirvLoweringOptions::default(),
        &mut actual,
    )
    .unwrap();
    assert_eq!(profile.binding_bytes[3], profile.output_count * 12);
    if actual != expected {
        *failures += 1;
        eprintln!(
            "encoded mismatch {scalar:?} {operation:?} {policy:?}: differing words={}",
            actual.iter().zip(&expected).filter(|(a, b)| a != b).count()
        );
    }
}
fn verify(failures: &mut usize) {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for (policy, encoded_policy) in [
            (PcuFloatUnderflowPolicy::IeeeAfterRounding, 0),
            (PcuFloatUnderflowPolicy::RejectSubnormalResult, 1),
            (PcuFloatUnderflowPolicy::AllowGradualUnderflow, 2),
        ] {
            for (count, f32_bits, f64_bits) in [
                (3, 0x4040_0000_u32, 0x4008_0000_0000_0000_u64),
                ((1 << 24) + 1, 0x4b80_0000, 0x4170_0000_1000_0000),
            ] {
                let factor = if scalar == PcuScalarType::F32 {
                    u64::from(f32_bits)
                } else {
                    f64_bits
                };
                let bytes = factor.to_le_bytes();
                let low = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                let high = u32::from_le_bytes(bytes[4..].try_into().unwrap());
                compare(
                    scalar,
                    PcuSpirvCompoundOperation::MeanSquaredError {
                        count: black_box(count),
                    },
                    [2, encoded_policy, 1, count, 1, 0, 0, low, high],
                    policy,
                    failures,
                );
            }
            for (f32_bits, f64_bits) in [
                (0, 0_u64),
                (0x8000_0000, 0x8000_0000_0000_0000),
                (1, 0x36a0_0000_0000_0000),
                (0x8000_0001, 0xb6a0_0000_0000_0000),
                (0x3f80_0000, 0x3ff0_0000_0000_0000),
                (0x7f7f_ffff, 0x47ef_ffff_e000_0000),
            ] {
                let factor = if scalar == PcuScalarType::F32 {
                    u64::from(f32_bits)
                } else {
                    f64_bits
                };
                let bytes = factor.to_le_bytes();
                let low = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                let high = u32::from_le_bytes(bytes[4..].try_into().unwrap());
                compare(
                    scalar,
                    PcuSpirvCompoundOperation::Sgd {
                        count: 65,
                        learning_rate: black_box(f32::from_bits(black_box(f32_bits))),
                    },
                    [0, encoded_policy, 1, 65, 1, 0, 0, low, high],
                    policy,
                    failures,
                );
            }
            for bits in [0x7f80_0000, 0x7fc0_0001] {
                let mut words = Vec::new();
                assert_eq!(
                    lower_checked_compound_to_spirv(
                        scalar,
                        PcuSpirvCompoundOperation::Sgd {
                            count: 65,
                            learning_rate: black_box(f32::from_bits(black_box(bits)))
                        },
                        policy,
                        PcuSpirvLoweringOptions::default(),
                        &mut words
                    ),
                    Err(PcuSpirvError::InvalidBinding)
                );
                assert!(words.is_empty());
            }
        }
    }
}
#[test]
fn encoded_constants_ignore_rounding_and_flush() {
    let original = environment::state();
    let mut failures = 0;
    for rounding in 0..4 {
        for flush in [false, true] {
            {
                let _guard = environment::Guard::enter(rounding, flush);
                eprintln!(
                    "encoded state rounding={rounding} flush={flush} {:?}",
                    environment::state()
                );
                verify(&mut failures);
            }
            assert_eq!(environment::state(), original);
        }
    }
    assert_eq!(
        failures, 0,
        "encoded source constants differ from independent literal metadata goldens"
    );
}

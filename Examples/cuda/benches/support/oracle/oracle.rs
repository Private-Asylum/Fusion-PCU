//! Exact separate-operation reference and deterministic single-invocation fault.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};

pub fn input(n: usize, phase: u16) -> Vec<f32> {
    (0..n)
        .map(|id| f32::from(u16::try_from(id % 251).unwrap()) + f32::from(phase))
        .collect()
}

pub fn bytes(input: &[f32]) -> Vec<u8> {
    input.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[allow(clippy::suboptimal_flops)] // The checked source evaluates multiply then add separately.
pub fn verify(input: &[f32], output: &[f32]) {
    assert_eq!(input.len(), output.len());
    for (input, actual) in input.iter().zip(output) {
        assert_eq!((input * 2.0 + 1.0).to_bits(), actual.to_bits());
    }
}

#[allow(clippy::suboptimal_flops)] // Match separate checked operations without allocating.
pub fn verify_bytes(input: &[f32], output: &[u8]) {
    let (words, remainder) = output.as_chunks::<4>();
    assert!(remainder.is_empty());
    assert_eq!(words.len(), input.len());
    for (input, word) in input.iter().zip(words) {
        assert_eq!((input * 2.0 + 1.0).to_bits(), u32::from_le_bytes(*word));
    }
}

pub const fn fault() -> PcuExecutionFault {
    PcuExecutionFault {
        kind: PcuExecutionFaultKind::InvalidFloatingOperand,
        invocation_id: 0,
        recovered: false,
    }
}

pub fn verify_fault(error: &PcuExecutionError) {
    assert!(matches!(error, PcuExecutionError::ArithmeticFault(actual) if *actual == fault()));
}

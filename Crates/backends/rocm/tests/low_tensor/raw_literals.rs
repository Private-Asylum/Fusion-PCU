//! Raw immutable data preserves encodings without inventing arithmetic support.
//!
//! IEEE 754-2019 floating encodings may contain signed zeros, NaNs and
//! subnormals. Transport does not evaluate them; checked arithmetic separately
//! rejects invalid operands. BF16 and OCP OFP8 are their own named encodings.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuCompoundArithmeticPolicy,
};

macro_rules! sources {
    ($module:ident, $scalar:ty, $one:expr, $two:expr, $four:expr, $maximum:expr, $payload:expr) => {
        mod $module {
            use super::*;
            pub const PAYLOAD: [$scalar; 3] = $payload;
            pub const ONE: $scalar = <$scalar>::from_bits($one);
            pub const FOUR: $scalar = <$scalar>::from_bits($four);
            pub const MAXIMUM: $scalar = <$scalar>::from_bits($maximum);
            #[pcu(crate_path=::pcu_facade)]
            pub fn pipeline(
                anchor: &[$scalar; 3],
            ) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                let literal = pcu::constant(const { [<$scalar>::from_bits($one); 3] });
                let uniform = pcu::uniform_like(anchor, const { <$scalar>::from_bits($two) });
                let sum = anchor + literal;
                Ok(sum * uniform)
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn mixed_fault(
                anchor: &[$scalar; 3],
            ) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                let literal = pcu::constant(const { [<$scalar>::from_bits($maximum); 3] });
                let uniform = pcu::uniform_like(anchor, const { <$scalar>::from_bits($two) });
                let sum = anchor + literal;
                Ok(sum * uniform)
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn discarded_nonfinite(
                anchor: &[$scalar; 3],
            ) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                let literal = pcu::constant(const { PAYLOAD });
                let _checked = anchor + literal;
                pcu::identity(anchor)
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn literal(_unused: &[$scalar]) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                pcu::constant(const { PAYLOAD })
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn uniform(anchor: &[$scalar; 3]) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                pcu::uniform_like(anchor, const { PAYLOAD[0] })
            }
        }
    };
}

sources!(
    binary16,
    PcuF16Bits,
    0x3c00,
    0x4000,
    0x4400,
    0x7bff,
    [
        PcuF16Bits::from_bits(0x7c42),
        PcuF16Bits::from_bits(0x8000),
        PcuF16Bits::from_bits(1)
    ]
);
sources!(
    brain16,
    PcuBf16Bits,
    0x3f80,
    0x4000,
    0x4080,
    0x7f7f,
    [
        PcuBf16Bits::from_bits(0x7f82),
        PcuBf16Bits::from_bits(0x8000),
        PcuBf16Bits::from_bits(1)
    ]
);
sources!(
    e4m3fn,
    PcuF8E4M3FnBits,
    0x38,
    0x40,
    0x48,
    0x7e,
    [
        PcuF8E4M3FnBits::from_bits(0x7f),
        PcuF8E4M3FnBits::from_bits(0x80),
        PcuF8E4M3FnBits::from_bits(1)
    ]
);
sources!(
    e5m2,
    PcuF8E5M2Bits,
    0x3c,
    0x40,
    0x44,
    0x7b,
    [
        PcuF8E5M2Bits::from_bits(0x7d),
        PcuF8E5M2Bits::from_bits(0x80),
        PcuF8E5M2Bits::from_bits(1)
    ]
);

#[pcu(crate_path=::pcu_facade, invocations = 3)]
fn overwrite<T: PcuScalar>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}
#[pcu(crate_path=::pcu_facade)]
fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
fn check<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T; 3]) {
    let mut stack = [expected[1]; 5];
    owner.read_into(&mut stack).unwrap();
    for (actual, expected) in stack[..3].iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    for tail in &stack[3..] {
        assert_eq!(tail.encode_le().as_ref(), expected[1].encode_le().as_ref());
    }
}
macro_rules! verify {
    ($module:ident) => {{
        let mut literal = $module::literal(&[]).unwrap();
        check(&literal, &$module::PAYLOAD);
        let uniform = $module::uniform(&$module::PAYLOAD).unwrap();
        check(&uniform, &[$module::PAYLOAD[0]; 3]);
        // Mutating a returned owner must never overwrite a cached constant.
        overwrite(&[$module::PAYLOAD[1]], &mut literal).unwrap();
        let mutated = consume(literal).unwrap();
        check(&mutated, &[$module::PAYLOAD[1]; 3]);
        let replay = $module::literal(&[]).unwrap();
        check(&replay, &$module::PAYLOAD);
        global::clear_thread_cache().unwrap();
        check(&replay, &$module::PAYLOAD);
        check(&uniform, &[$module::PAYLOAD[0]; 3]);
        check(&mutated, &[$module::PAYLOAD[1]; 3]);
    }};
}
#[test]
#[ignore = "actual ROCm raw literal transport and fresh owned outputs"]
fn four_low_carriers_preserve_raw_literal_bits_across_policies_and_ownership() {
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for float_underflow in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                global::configure(global::PcuExecutionPolicy {
                    backend: global::PcuBackendChoice::Rocm,
                    device: Some(0),
                    float_underflow,
                    numerical_options: PcuNumericalOptions {
                        precision,
                        compound_arithmetic,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .unwrap();
                global::clear_thread_cache().unwrap();
                verify!(binary16);
                verify!(brain16);
                verify!(e4m3fn);
                verify!(e5m2);
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

macro_rules! math {
    ($module:ident) => {{
        let healthy = [$module::ONE; 3];
        check(&$module::pipeline(&healthy).unwrap(), &[$module::FOUR; 3]);
        assert!(matches!($module::discarded_nonfinite(&healthy), Err(PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id == 0 && fault.kind == fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand && !fault.recovered));
        // A later Add fault precedes an earlier invocation's Mul fault.
        let mixed = [$module::ONE, $module::MAXIMUM, $module::ONE];
        assert!(matches!($module::mixed_fault(&mixed), Err(PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id == 1 && fault.kind == fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow && !fault.recovered));
        check(&$module::pipeline(&healthy).unwrap(), &[$module::FOUR; 3]);
    }};
}
#[test]
#[ignore = "actual ROCm low literal-fed checked Add/Mul and discarded effects"]
fn four_low_literal_math_retains_operation_fault_order_and_retry() {
    for numerical_mode in [
        fusion_pcu::PcuNumericalMode::Boundary,
        fusion_pcu::PcuNumericalMode::Strict,
    ] {
        for float_underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Rocm,
                device: Some(0),
                numerical_mode,
                float_underflow,
                ..Default::default()
            })
            .unwrap();
            global::clear_thread_cache().unwrap();
            math!(binary16);
            math!(brain16);
            math!(e4m3fn);
            math!(e5m2);
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

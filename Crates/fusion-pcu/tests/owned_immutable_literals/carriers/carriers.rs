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
    PcuF128Bits,
    PcuF256Bits,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuCompoundArithmeticPolicy,
};

macro_rules! sources {
    ($module:ident, $scalar:ty, $payload:expr) => {
        mod $module {
            use super::*;
            pub const PAYLOAD: [$scalar; 3] = $payload;
            #[pcu]
            pub fn literal(_unused: &[$scalar]) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                pcu::constant(const { PAYLOAD })
            }
            #[pcu]
            pub fn uniform(anchor: &[$scalar; 3]) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                pcu::uniform_like(anchor, const { PAYLOAD[0] })
            }
            #[pcu]
            pub fn zero_argument_literal() -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                pcu::constant(const { PAYLOAD })
            }
            #[pcu]
            pub fn zero_argument_uniform() -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                let anchor = pcu::constant(const { PAYLOAD })?;
                pcu::uniform_like(&anchor, const { PAYLOAD[0] })
            }
            #[pcu]
            pub fn matrix() -> Result<PcuTensor<$scalar>, PcuExecutionError> {
                pcu::constant(const { [PAYLOAD, [PAYLOAD[2], PAYLOAD[1], PAYLOAD[0]]] })
            }
        }
    };
}
sources!(
    binary32,
    f32,
    [
        f32::from_bits(0x7f80_0042),
        f32::from_bits(0x8000_0000),
        f32::from_bits(1)
    ]
);
sources!(
    binary64,
    f64,
    [
        f64::from_bits(0x7ff0_0000_0000_0042),
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(1)
    ]
);
sources!(
    binary16,
    PcuF16Bits,
    [
        PcuF16Bits::from_bits(0x7c42),
        PcuF16Bits::from_bits(0x8000),
        PcuF16Bits::from_bits(1)
    ]
);
sources!(
    brain16,
    PcuBf16Bits,
    [
        PcuBf16Bits::from_bits(0x7f82),
        PcuBf16Bits::from_bits(0x8000),
        PcuBf16Bits::from_bits(1)
    ]
);
sources!(
    e4m3fn,
    PcuF8E4M3FnBits,
    [
        PcuF8E4M3FnBits::from_bits(0x7f),
        PcuF8E4M3FnBits::from_bits(0x80),
        PcuF8E4M3FnBits::from_bits(1)
    ]
);
sources!(
    e5m2,
    PcuF8E5M2Bits,
    [
        PcuF8E5M2Bits::from_bits(0x7d),
        PcuF8E5M2Bits::from_bits(0x80),
        PcuF8E5M2Bits::from_bits(1)
    ]
);
sources!(
    binary128,
    PcuF128Bits,
    [
        PcuF128Bits::from_limbs_le([0x42, 0x7fff_0000_0000_0000]),
        PcuF128Bits::from_limbs_le([0, 0x8000_0000_0000_0000]),
        PcuF128Bits::from_limbs_le([1, 0])
    ]
);
sources!(
    binary256,
    PcuF256Bits,
    [
        PcuF256Bits::from_limbs_le([0x42, 0, 0, 0x7fff_f000_0000_0000]),
        PcuF256Bits::from_limbs_le([0, 0, 0, 0x8000_0000_0000_0000]),
        PcuF256Bits::from_limbs_le([1, 0, 0, 0])
    ]
);

#[pcu(invocations = 3)]
fn overwrite<T: PcuScalar>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}
#[pcu]
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

fn check_matrix<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[[T; 3]; 2]) {
    assert_eq!(owner.shape(), [2, 3]);
    let mut stack = [expected[0][1]; 8];
    owner.read_into(&mut stack).unwrap();
    for (actual, expected) in stack[..6].iter().zip(expected.iter().flatten()) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    for tail in &stack[6..] {
        assert_eq!(
            tail.encode_le().as_ref(),
            expected[0][1].encode_le().as_ref()
        );
    }
    let mut short = [expected[0][1]; 5];
    assert!(owner.read_into(&mut short).is_err());
    for value in short {
        assert_eq!(
            value.encode_le().as_ref(),
            expected[0][1].encode_le().as_ref()
        );
    }
}

macro_rules! verify_matrix {
    ($module:ident) => {{
        let expected = [
            $module::PAYLOAD,
            [
                $module::PAYLOAD[2],
                $module::PAYLOAD[1],
                $module::PAYLOAD[0],
            ],
        ];
        let mut owner = $module::matrix().unwrap();
        check_matrix(&owner, &expected);
        let replacement = [[$module::PAYLOAD[1]; 3]; 2];
        overwrite_matrix(&replacement, &mut owner).unwrap();
        let consumed = consume(owner).unwrap();
        check_matrix(&consumed, &replacement);
        let replay = $module::matrix().unwrap();
        check_matrix(&replay, &expected);
        global::clear_thread_cache().unwrap();
        check_matrix(&replay, &expected);
        check_matrix(&consumed, &replacement);
    }};
}
macro_rules! verify {
    ($module:ident) => {{
        let uniforms = [
            $module::uniform(&$module::PAYLOAD).unwrap(),
            $module::zero_argument_uniform().unwrap(),
        ];
        for uniform in &uniforms {
            check(uniform, &[$module::PAYLOAD[0]; 3]);
        }
        // Both a pruned Rust argument and a genuine no-argument function must
        // retain independent escaped storage without fabricating an input.
        for mut literal in [
            $module::literal(&[]).unwrap(),
            $module::zero_argument_literal().unwrap(),
        ] {
            check(&literal, &$module::PAYLOAD);
            // Mutating a returned owner must never overwrite a cached constant.
            overwrite(&[$module::PAYLOAD[1]], &mut literal).unwrap();
            let mutated = consume(literal).unwrap();
            check(&mutated, &[$module::PAYLOAD[1]; 3]);
            let replays = [
                $module::literal(&[]).unwrap(),
                $module::zero_argument_literal().unwrap(),
            ];
            for replay in &replays {
                check(replay, &$module::PAYLOAD);
            }
            global::clear_thread_cache().unwrap();
            for replay in &replays {
                check(replay, &$module::PAYLOAD);
            }
            for uniform in &uniforms {
                check(uniform, &[$module::PAYLOAD[0]; 3]);
            }
            check(&mutated, &[$module::PAYLOAD[1]; 3]);
        }
    }};
}
// Shared native acceptance fixture: no provider-specific staging or upload API.
#[allow(clippy::redundant_pub_crate)] // Only the containing Cargo test registry may invoke this fixture.
pub(super) fn run(backend: global::PcuBackendChoice) {
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
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
                        backend,
                        numerical_mode,
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
                    verify!(binary32);
                    verify!(binary64);
                    verify!(binary128);
                    verify!(binary256);
                    verify_matrix!(binary16);
                    verify_matrix!(brain16);
                    verify_matrix!(e4m3fn);
                    verify_matrix!(e5m2);
                    verify_matrix!(binary32);
                    verify_matrix!(binary64);
                    verify_matrix!(binary128);
                    verify_matrix!(binary256);
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[pcu(invocations = 6)]
fn overwrite_matrix<T: PcuScalar>(input: &[[T; 3]; 2], output: &mut [[T; 3]; 2]) {
    let id = pcu::context::global_invocation_id();
    let row = id / 3;
    let column = id % 3;
    output[row][column] = input[row][column];
}

//! Genuine automatic low-format graph staging, escaping MLX owners and checked unused effects.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn relu<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}

#[pcu]
fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn unused_checked<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _checked = pcu::relu(input)?;
    pcu::identity(input)
}

#[pcu(flag(reject_subnormal_result))]
fn local_reject<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}

#[pcu(flag(allow_gradual_underflow))]
fn local_allow<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}

mod native {
    use super::*;
    #[rustfmt::skip]
    use fusion_pcu::{
        global,
        PcuBf16Bits,
        PcuCompoundArithmeticPolicy,
        PcuExecutionFaultKind,
        PcuF16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        PcuFloatUnderflowPolicy,
        PcuNumericalMode,
        PcuNumericalOptions,
        PcuPrecisionPolicy,
    };

    fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
        }
    }

    fn read<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T; 5], sentinel: T) {
        assert_eq!(owner.shape(), &[5]);
        let mut stack = [sentinel; 7];
        owner.read_into(&mut stack).unwrap();
        bits(&stack[..5], expected);
        bits(&stack[5..], &[sentinel; 2]);
    }

    fn fault<T: PcuScalar>(
        result: Result<PcuTensor<T>, PcuExecutionError>,
        kind: PcuExecutionFaultKind,
    ) {
        let Err(error) = result else {
            panic!("checked fault returned a usable owner")
        };
        assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
            if fault.kind == kind && fault.invocation_id == 2 && !fault.recovered));
    }

    macro_rules! format {
        ($ty:ty, $sign:expr, $normal:expr, $invalid:expr) => {{
            global::clear_thread_cache().unwrap();
            for phase in [0, 1, 2] {
                let raw = [$normal + phase, $sign | $normal, 0, $sign, $normal + 3];
                let input = raw.map(<$ty>::from_bits);
                let expected = [raw[0], 0, 0, 0, raw[4]].map(<$ty>::from_bits);
                let sentinel = <$ty>::from_bits($normal);
                let retained = retain::<$ty>(&input).unwrap();
                let activated = relu::<$ty>(&retained).unwrap();
                let unchanged = unused_checked::<$ty>(&retained).unwrap();
                read(&retained, &input, sentinel);
                read(&unchanged, &input, sentinel);
                read(&activated, &expected, sentinel);
                drop(retained);
                drop(unchanged);
                let moved = consume(activated).unwrap();
                read(&moved, &expected, sentinel);

                let mut invalid = input;
                invalid[2] = <$ty>::from_bits($invalid);
                let invalid_owner = retain::<$ty>(&invalid).unwrap();
                fault(
                    relu::<$ty>(&invalid_owner),
                    PcuExecutionFaultKind::InvalidFloatingOperand,
                );
                fault(
                    unused_checked::<$ty>(&invalid_owner),
                    PcuExecutionFaultKind::InvalidFloatingOperand,
                );
                read(&invalid_owner, &invalid, sentinel);
                read(&moved, &expected, sentinel);

                // Function-local policy wins over either global underflow default.
                let mut tiny = input;
                tiny[2] = <$ty>::from_bits(1);
                fault(
                    local_reject::<$ty>(&tiny),
                    PcuExecutionFaultKind::ArithmeticUnderflow,
                );
                let accepted = local_allow::<$ty>(&tiny).unwrap();
                let mut tiny_expected = expected;
                tiny_expected[2] = tiny[2];
                read(&accepted, &tiny_expected, sentinel);
                let retry = relu::<$ty>(&input).unwrap();
                read(&retry, &expected, sentinel);
            }
        }};
    }

    #[allow(clippy::cognitive_complexity)] // Four exact representation witnesses share one policy/liveness matrix.
    fn formats() {
        format!(PcuF16Bits, 0x8000, 0x0400, 0x7c00);
        format!(PcuBf16Bits, 0x8000, 0x0080, 0x7f80);
        format!(PcuF8E4M3FnBits, 0x80, 0x08, 0x7f);
        format!(PcuF8E5M2Bits, 0x80, 0x04, 0x7c);
    }

    pub fn verify() {
        let _guard = super::super::POLICY_LOCK.lock().unwrap();
        for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            for compound_arithmetic in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    for float_underflow in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    ] {
                        global::configure(global::PcuExecutionPolicy {
                            backend: global::PcuBackendChoice::Mlx,
                            numerical_mode,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                ..Default::default()
                            },
                            float_underflow,
                            ..Default::default()
                        })
                        .unwrap();
                        formats();
                    }
                }
            }
        }
        global::clear_thread_cache().unwrap();
        global::use_defaults().unwrap();
    }
}

pub fn verify() {
    native::verify();
}

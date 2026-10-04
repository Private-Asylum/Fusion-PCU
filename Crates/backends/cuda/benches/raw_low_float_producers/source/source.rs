//! Inline-const raw float encodings; the payload is never evaluated as arithmetic.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuScalar,
    PcuTensor,
};
use fusion_pcu::dialect::tensor::TensorElement;
pub trait RawSource: PcuScalar + TensorElement {
    const LABEL: &'static str;
    const SENTINEL: Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn literal<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn uniform<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn capture<const N: usize>(
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        underflow: PcuFloatUnderflowPolicy,
    );
}
macro_rules! carrier {
    ($scalar:ty, $label:literal, $sentinel:expr, $literal:ident, $uniform:ident, $matrix:ident, $matrix_uniform:ident) => {
        impl RawSource for $scalar {
            const LABEL: &'static str = $label;
            const SENTINEL: Self = $sentinel;
            fn from_bytes(bytes: &[u8]) -> Self {
                Self::decode_le(bytes.try_into().unwrap())
            }
            fn literal<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError> {
                if N == 6 { $matrix() } else { $literal::<N>() }
            }
            fn uniform<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError> {
                if N == 6 {
                    $matrix_uniform()
                } else {
                    $uniform::<N>()
                }
            }
            fn capture<const N: usize>(
                mode: PcuNumericalMode,
                options: PcuNumericalOptions,
                underflow: PcuFloatUnderflowPolicy,
            ) {
                let builders = if N == 6 {
                    [
                        $matrix::__pcu_capture_entry,
                        $matrix_uniform::__pcu_capture_entry,
                    ]
                } else {
                    [
                        $literal::__pcu_capture_entry::<N>,
                        $uniform::__pcu_capture_entry::<N>,
                    ]
                };
                let expected_shape = if N == 6 { vec![2, 3] } else { vec![N] };
                for builder in builders {
                    let captured = global::__pcu_capture_tensor_program::<Self, 0, _>(
                        [],
                        underflow,
                        mode,
                        options,
                        builder,
                    )
                    .unwrap();
                    assert!(captured.argument_indices().is_empty());
                    assert!(captured.program().input_values().is_empty());
                    for &value in captured.program().selected_nodes() {
                        let node = captured.program().graph().node(value).unwrap();
                        assert_eq!(node.numerical_mode, None);
                        assert_eq!(node.float_underflow_policy, None);
                        assert_eq!(node.numerical_options, options);
                        assert_eq!(node.shape, expected_shape);
                    }
                }
            }
        }
    };
}
macro_rules! width {
    ($module:ident,$ty:ty,$label:literal,$patterns:expr,$sentinel:expr) => {
        mod $module {
            use super::*;
            const fn payload<const N: usize>() -> [$ty; N] {
                let patterns = $patterns;
                let mut result = [<$ty>::from_bits(0); N];
                let mut index = 0;
                while index < N {
                    result[index] = <$ty>::from_bits(patterns[index % 6]);
                    index += 1;
                }
                result
            }
            #[pcu]
            fn literal<const N: usize>() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                pcu::constant(const { payload::<N>() })
            }
            #[pcu]
            fn uniform<const N: usize>() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                let shape = pcu::constant(const { payload::<N>() })?;
                pcu::uniform_like(&shape, const { payload::<1>()[0] })
            }
            #[pcu]
            fn matrix() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                pcu::constant(
                    const {
                        [
                            [payload::<6>()[0], payload::<6>()[1], payload::<6>()[2]],
                            [payload::<6>()[3], payload::<6>()[4], payload::<6>()[5]],
                        ]
                    },
                )
            }
            #[pcu]
            fn matrix_uniform() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                let shape = matrix()?;
                pcu::uniform_like(&shape, const { payload::<1>()[0] })
            }
            carrier!(
                $ty,
                $label,
                $sentinel,
                literal,
                uniform,
                matrix,
                matrix_uniform
            );
        }
    };
}
width!(
    binary16,
    PcuF16Bits,
    "f16",
    [0x7e42, 0x8000, 1, 0xfe43, 0x7bff, 0x7c00],
    PcuF16Bits::from_bits(0x3c00)
);
width!(
    brain16,
    PcuBf16Bits,
    "bf16",
    [0x7fc2, 0x8000, 1, 0xffc3, 0x7f7f, 0x7f80],
    PcuBf16Bits::from_bits(0x3f80)
);
width!(
    e4m3fn,
    PcuF8E4M3FnBits,
    "e4m3fn",
    [0x7f, 0x80, 1, 0xff, 0x7e, 7],
    PcuF8E4M3FnBits::from_bits(0x38)
);
width!(
    e5m2,
    PcuF8E5M2Bits,
    "e5m2",
    [0x7f, 0x80, 1, 0xff, 0x7b, 0x7c],
    PcuF8E5M2Bits::from_bits(0x3c)
);
#[pcu(invocations = N)]
pub fn overwrite<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let index = pcu::context::global_invocation_id();
    output[index] = input[0];
}
#[pcu]
pub fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(invocations = 6)]
pub fn overwrite_matrix<T: PcuScalar>(input: &[[T; 3]; 2], output: &mut [[T; 3]; 2]) {
    let id = pcu::context::global_invocation_id();
    let row = id / 3;
    let column = id % 3;
    output[row][column] = input[row][column];
}
